#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


__global__ void sync_conv_groups_softmax_with_loss() 
{     }

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    //  图片编号
    const int n = index / spatial_dim;
    //  0
    const int s = index % spatial_dim;
    //  
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) 
    {
      loss[index] = 0;
      counts[index] = 0;
    } 
    else 
    {
      //prob_dat[n*1000+label]
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // 得到前向的值
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  // 标签的值
  const Dtype* label = bottom[1]->gpu_data();
  // 得到C*H*W
  const int dim = prob_.count() / outer_num_;
  // 同样是只有N线程
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything, we use it here to avoid having
  // to allocate new GPU memory to accumulate intermediate results.
  // 得到损失的值。 
  // 将每一个样本的损失的值暂时借用bottom[0]->mutable_gpu_diff()的空间存储
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  // 每一张图片的损失在loss里面。
  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS,0,stream_[0]>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
  // 定义在CPU段的。
  Dtype loss;
  // 对于损失的值进行相加
  caffe_gpu_asum1(nthreads, loss_data, &loss,handle_[0]);
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  // 统计有效的样本数目
  if (normalization_ == LossParameter_NormalizationMode_VALID &&has_ignore_label_) 
  {
    caffe_gpu_asum1(nthreads, counts, &valid_count,handle_[0]);
  }
  //top[0]->mutable_cpu_data()[0]  第一个值记录着损失的和
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                      valid_count);
  // top[1]的指针指向prob_
  if (top.size() == 2) 
  {
    top[1]->ShareData(prob_);
  }
  // Clear scratch memory to prevent interfering with backward (see #6202).
  // bottom[0]->mutable_gpu_diff()进行清空，因为反向是需要使用。
  caffe_gpu_set1(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff(),stream_[0]);
  //  这里面应该需要进行流的同步问题。
  //  其实并不需要的。sync_conv_groups_softmax_with_loss<<<1, 1>>>();

}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  // 1000
  const int channels = dim / spatial_dim;
  //LOG(INFO)<<"spatial_dim   "<<spatial_dim<<"dim  "<<dim <<" channels"<<channels;
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    //  还是n      spatial_dim=1
    const int n = index / spatial_dim;
    //  0
    const int s = index % spatial_dim;
    //   得到其实还是label[n]
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    // 假设这个标签呼略，这个样本的梯度都是0
    if (has_ignore_label_ && label_value == ignore_label_)
     {
      // 内部通道数目
      for (int c = 0; c < channels; ++c) 
      {
        // bottom[n*1000+c]
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      // 表示这个样本被忽略
      counts[index] = 0;
    } 
    else 
    {
      // bottom_diff[n*1000+label]-=1,其他保持不变。
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) 
  {
    //  需要计算的梯度
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    //  得到概率值
    const Dtype* prob_data = prob_.gpu_data();
    //  得到的总的损失。
    //  
    const Dtype* top_data = top[0]->gpu_data();
    //  prob_data-----存在 bottom_diff。
    //  由于求导之后，等于prob_data 或者prob_data-1
    //  因而直接判断进行加一减去1即可。
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    // 获取标签
    const Dtype* label = bottom[1]->gpu_data();
    // 等于 C*H*W q 其中H=W=1 
    // 假设输入为  10*1000*1*1
    
    //  prob_.count()=10*1000
    // dim =C*H*W
    const int dim = prob_.count() / outer_num_;
     //LOG(INFO)<<"outer_num_  "<<outer_num_<<"  inner_num_  "<<inner_num_<<"  dim  "<<dim<<"  prob_.count()  "<<prob_.count();
    // 等于N*C*H*W  其实就是N 
    //  10个线程而已。

    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    //  借用pro的存储空间，作为中间值
    // 太浪费了，一个是  10*1000  一个是10*1  虽然重用
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS,0,stream_[0]>>>(nthreads, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts);
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) 
    {
      //  计算 vector x 的所有element的绝对值之和。
      caffe_gpu_asum1(nthreads, counts, &valid_count,handle_[0]);
    }
      //  
      //  得到一个常数的值。
      
      //  得到归一化德损失，实际是所有样本的损失的和，除以有效样本数目。
    //LOG(INFO)<<" top[0]->cpu_diff()[0] "<< top[0]->cpu_diff()[0]<<" count   "<<valid_count;  
    const Dtype loss_weight = top[0]->cpu_diff()[0] /
                              get_normalizer(normalization_, valid_count);
    //     获得。。。。。。 y = alpha*x 
    // LOG(INFO)<<"  loss_weight  "<<loss_weight;
    // LOG(INFO)<<"  top[0]->cpu_diff()[0]"<<top[0]->cpu_diff()[0];
    // LOG(INFO)<<"  get_normalizer(normalization_, valid_count)  "<<get_normalizer(normalization_, valid_count);
    caffe_gpu_scal1(prob_.count(), loss_weight , bottom_diff,handle_[0]);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
