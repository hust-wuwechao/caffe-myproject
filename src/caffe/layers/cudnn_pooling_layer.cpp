#ifdef USE_CUDNN
#include <vector>

#include "caffe/layers/cudnn_pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);
  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createPoolingDesc<Dtype>(&pooling_desc_,
      this->layer_param_.pooling_param().pool(), &mode_,
      this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
      this->stride_h_, this->stride_w_);
  handles_setup_ = true;
}


void CuDNNPoolingLayer<Dtype>::LayerSetUp1(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top,
    cudnnHandle_t* handle ,
    cudaStream_t*  stream ) 
{
  PoolingLayer<Dtype>::LayerSetUp(bottom, top);
  //CUDNN_CHECK(cudnnCreate(&handle_));
  handle_=handle;
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createPoolingDesc<Dtype>(
      &pooling_desc_,
      this->layer_param_.pooling_param().pool(),
      &mode_,
      this->kernel_h_, this->kernel_w_, this->pad_h_, this->pad_w_,
      this->stride_h_, this->stride_w_);
  handles_setup_ = true;
}





template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  PoolingLayer<Dtype>::Reshape(bottom, top);
   // 这里面仍然是调用父类的方法。 
   cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, bottom[0]->num(),
       this->channels_, this->height_, this->width_);
   cudnn::setTensor4dDesc<Dtype>(&top_desc_, bottom[0]->num(),
       this->channels_, this->pooled_height_, this->pooled_width_);
}

template <typename Dtype>
CuDNNPoolingLayer<Dtype>::~CuDNNPoolingLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyPoolingDescriptor(pooling_desc_);
  //cudnnDestroy(handle_);
}

INSTANTIATE_CLASS(CuDNNPoolingLayer);

}   // namespace caffe
#endif
