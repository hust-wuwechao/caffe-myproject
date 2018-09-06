#include <vector>

#include "caffe/layers/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#define CUDNN_STREAMS_PER_GROUP 3
#define GROUP 1
template <typename Dtype>
void SplitLayer<Dtype>::LayerSetUp1(
     const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,
      cudnnHandle_t*  handle,
      cudaStream_t*   stream) 
{


}

template <typename Dtype>
void SplitLayer<Dtype>::LayerSetUp1(
     const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,
      cudnnHandle_t*  handle,
      cudaStream_t*   stream) 
{
  stream_=stream;
  handle_=new cublasHandle_t[GROUP*CUDNN_STREAMS_PER_GROUP];
  // 自己创建handle
  // 并且和和流进行绑定
  // 我们仍然创建3个。
  // 其中第一个为优先级最高
  // 第二，3个问优先级最低
  for(int i=0;i<1;i++)
  {
    cublasCreate(&handle_[i]);
    cublasSetStream(handle_[i],  stream_[i]);
  }
}   



template <typename Dtype>
void SplitLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  count_ = bottom[0]->count();
  for (int i = 0; i < top.size(); ++i) 
  {
    // Do not allow in-place computation in the SplitLayer.  Instead, share data
    // by reference in the forward pass, and keep separate diff allocations in
    // the backward pass.  (Technically, it should be possible to share the diff
    // blob of the first split output with the input, but this seems to cause
    // some strange effects in practice...)
    CHECK_NE(top[i], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    top[i]->ReshapeLike(*bottom[0]);
    CHECK_EQ(count_, top[i]->count());
  }
}

template <typename Dtype>
void SplitLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  for (int i = 0; i < top.size(); ++i)
  {
     top[i]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void SplitLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  if (top.size() == 1) {
    caffe_copy(count_, top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
    return;
  }
  //  将多个的梯度进行累加的方式。
  
  caffe_add(count_, top[0]->cpu_diff(), top[1]->cpu_diff(),
            bottom[0]->mutable_cpu_diff());
  // Add remaining top blob diffs.
  // 如果会被split成多份的话
  for (int i = 2; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_axpy(count_, Dtype(1.), top_diff, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(SplitLayer);
#endif

INSTANTIATE_CLASS(SplitLayer);
REGISTER_LAYER_CLASS(Split);

}  // namespace caffe
