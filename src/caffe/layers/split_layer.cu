#include <vector>

#include "caffe/layers/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
  {
    //这里面需要同步流吗？？？？
     // 我觉得而不需要的
     // 我们加一下哈。
     //
    //  先进行stream的同步。
    //  stream[0]和stream[3]
    //  接下来stream[0]等待这2个完成
    //  只需要加一个的事件等待。
    //  声明
    cudaEvent_t event;
    //  创建
    cudaEventCreate(&event);
    cudaEventRecord(event,stream_[0]);
    //cudaStreamWaitEvent(stream_[0],event);
    cudaEventSynchronize(event);
    //  这样完成了流之间的同步的过程。
    //  接下来都是在流1里面完成的。
  for (int i = 0; i < top.size(); ++i)
  {
  
    top[i]->ShareData(*bottom[0]);
  }
}

template <typename Dtype>
void SplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) 
   { return; }
  if (top.size() == 1) 
  {
    caffe_copy(count_, top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
    return;
  }
   //这里面需要同步流吗？？？？
     // 我觉得而不需要的
     // 我们加一下哈。
     //
    //  先进行stream的同步。
    //  stream[0]和stream[3]
    //  接下来stream[0]等待这2个完成
    //  只需要加一个的事件等待。
    //  声明
    cudaEvent_t event;
    //  创建
    cudaEventCreate(&event);
    cudaEventRecord(event,stream_[3]);
    cudaStreamWaitEvent(stream_[0],event,0);
    //  cudaEventSynchronize(event);
    //  这样完成了流之间的同步的过程。
    //  接下来都是在流1里面完成的。
    //  top[0]->gpu_diff(), top[1]->gpu_diff()都完成之后。
  caffe_gpu_add1(count_, top[0]->gpu_diff(), top[1]->gpu_diff(),
                bottom[0]->mutable_gpu_diff(),stream_[0]);
  // Add remaining top blob diffs.
  // 只要2个一般
  for (int i = 2; i < top.size(); ++i)
  {
    const Dtype* top_diff = top[i]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_axpy1(count_, Dtype(1.), top_diff, bottom_diff,handle_[0]);
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(SplitLayer);

}  // namespace caffe
