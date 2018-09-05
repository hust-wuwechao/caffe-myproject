#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void BiasForward(const int n, const Dtype* in,
    const Dtype* bias, const int bias_dim, const int inner_dim,
    Dtype* out) {
  CUDA_KERNEL_LOOP(index, n)
   {
    // bias_dim等于通道数
    // bias_index得到属于的通道数
    // 对应相加即可了。
    
    const int bias_index = (index / inner_dim) % bias_dim;
    out[index] = in[index] + bias[bias_index];
  }
}

template <typename Dtype>
void BiasLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
 {
  const int count = top[0]->count();
  //N*C*H*W
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* bias_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->gpu_data();
  
  Dtype* top_data = top[0]->mutable_gpu_data();
  // 设置线程的个数为512个？
  // 这里面其实也有值得研究的地方
  // 也就是说这里面是所有的图片在一个的kernel里面执行的

  BiasForward<Dtype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS,0,stream_[0]>>>
      (
      count, bottom_data, bias_data, bias_dim_, inner_dim_, top_data
      );
}

template <typename Dtype>
void BiasLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  // 由于只是相加，所有梯度值应该相同。
  if (propagate_down[0] && bottom[0] != top[0]) 
  {
    LOG(INFO)<<"进入到bias 里面的反向过程并且部位原地的过程";
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    //  所以下面的不会执行。
    //  核心就是这一句
    //  
    caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  }
  // in-place, we don't need to do anything with the data diff
  // 我们需要计算blas 的参数值的diff 。
  // 这个自然也不处于关键路劲上。
  //  
  const bool bias_param = (bottom.size() == 1);
  if ((!bias_param && propagate_down[1]) ||
      (bias_param && this->param_propagate_down_[0])) 
  {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bias_diff = (bias_param ? this->blobs_[0].get() : bottom[1])
        ->mutable_gpu_diff();
    bool accum = bias_param;
    LOG(INFO)<<"outer_dim_ ======"<<outer_dim_;
    for (int n = 0; n < outer_dim_; ++n) 
    {
      // 对每一张图片进行scale的计算
      caffe_gpu_gemv1(CblasNoTrans, bias_dim_, inner_dim_, Dtype(1),
          top_diff, 
          bias_multiplier_.gpu_data(),
          Dtype(accum), 
          bias_diff, 
          handle_[1]);
      top_diff += dim_;
      accum = true;
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(BiasLayer);

}  // namespace caffe
