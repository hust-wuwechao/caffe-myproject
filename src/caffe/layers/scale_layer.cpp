#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#define CUDNN_STREAMS_PER_GROUP 3
#define GROUP 1
template <typename Dtype>
void ScaleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  if (bottom.size() == 1 && this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else if (bottom.size() == 1) {
    // scale is a learned parameter; initialize it
    axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
          << "scale blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis_;
    }
    this->blobs_.resize(1);
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + axis_;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> scale_shape(shape_start, shape_end);
    this->blobs_[0].reset(new Blob<Dtype>(scale_shape));
    FillerParameter filler_param(param.filler());
    if (!param.has_filler()) {
      // Default to unit (1) filler for identity operation.
      filler_param.set_type("constant");
      filler_param.set_value(1);
    }
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[0].get());
  }
  if (param.bias_term()) {
    LayerParameter layer_param(this->layer_param_);
    layer_param.set_type("Bias");
    BiasParameter* bias_param = layer_param.mutable_bias_param();
    bias_param->set_axis(param.axis());
    if (bottom.size() > 1) {
      bias_param->set_num_axes(bottom[1]->num_axes());
    } else {
      bias_param->set_num_axes(param.num_axes());
    }
    bias_param->mutable_filler()->CopyFrom(param.bias_filler());
    bias_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    bias_bottom_vec_.resize(1);
    bias_bottom_vec_[0] = bottom[0];
    bias_layer_->SetUp(bias_bottom_vec_, top);
    if (this->blobs_.size() + bottom.size() < 3) {
      // case: blobs.size == 1 && bottom.size == 1
      // or blobs.size == 0 && bottom.size == 2
      bias_param_id_ = this->blobs_.size();
      this->blobs_.resize(bias_param_id_ + 1);
      this->blobs_[bias_param_id_] = bias_layer_->blobs()[0];
    } else {
      // bias param already initialized
      bias_param_id_ = this->blobs_.size() - 1;
      bias_layer_->blobs()[0] = this->blobs_[bias_param_id_];
    }
    bias_propagate_down_.resize(1, false);
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void ScaleLayer<Dtype>::LayerSetUp1(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,
      cudaStream_t*    stream,
      cublasHandle_t*  handle) 
{
  const ScaleParameter& param = this->layer_param_.scale_param();
  stream_=stream;
  handle_=new cublasHandle_t[GROUP*CUDNN_STREAMS_PER_GROUP];
  // 自己创建handle
  // 并且和和流进行绑定
  // 我们仍然创建3个。
  // 其中第一个为优先级最高
  // 第二，3个问优先级最低
  for(int i=0;i<3;i++)
  {
    cublasCreate(&handle_[i]);
    cublasSetStream(handle_[i],  stream_[i]);
  }
  if (bottom.size() == 1 && this->blobs_.size() > 0) 
  {
    LOG(INFO) << "Skipping parameter initialization";
  } 
  else if 
  (bottom.size() == 1) 
  {
    // scale is a learned parameter; initialize it
    axis_ = bottom[0]->CanonicalAxisIndex(param.axis());
    const int num_axes = param.num_axes();
    CHECK_GE(num_axes, -1) << "num_axes must be non-negative, "
                           << "or -1 to extend to the end of bottom[0]";
    if (num_axes >= 0) {
      CHECK_GE(bottom[0]->num_axes(), axis_ + num_axes)
          << "scale blob's shape extends past bottom[0]'s shape when applied "
          << "starting with bottom[0] axis = " << axis_;
    }
    this->blobs_.resize(1);
    const vector<int>::const_iterator& shape_start =
        bottom[0]->shape().begin() + axis_;
    const vector<int>::const_iterator& shape_end =
        (num_axes == -1) ? bottom[0]->shape().end() : (shape_start + num_axes);
    vector<int> scale_shape(shape_start, shape_end);
    this->blobs_[0].reset(new Blob<Dtype>(scale_shape));
    FillerParameter filler_param(param.filler());
    if (!param.has_filler()) {
      // Default to unit (1) filler for identity operation.
      filler_param.set_type("constant");
      filler_param.set_value(1);
    }
    shared_ptr<Filler<Dtype> > filler(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[0].get());
  }
  if (param.bias_term()) {
    LayerParameter layer_param(this->layer_param_);
    layer_param.set_type("Bias");
    BiasParameter* bias_param = layer_param.mutable_bias_param();
    bias_param->set_axis(param.axis());
    if (bottom.size() > 1) {
      bias_param->set_num_axes(bottom[1]->num_axes());
    } 
    else 
    {
      bias_param->set_num_axes(param.num_axes());
    }
    bias_param->mutable_filler()->CopyFrom(param.bias_filler());
    bias_layer_ = LayerRegistry<Dtype>::CreateLayer(layer_param);
    bias_bottom_vec_.resize(1);
    bias_bottom_vec_[0] = bottom[0];
    // 这里面进行修改。我们bias的层传入哦们需要的结果。
    // 我们把bias 里面的需要的留传我们的需要的流。看一下结果。

    bias_layer_->SetUp1(bias_bottom_vec_, top,stream,handle);
    if (this->blobs_.size() + bottom.size() < 3) {
      // case: blobs.size == 1 && bottom.size == 1
      // or blobs.size == 0 && bottom.size == 2
      bias_param_id_ = this->blobs_.size();
      this->blobs_.resize(bias_param_id_ + 1);
      this->blobs_[bias_param_id_] = bias_layer_->blobs()[0];
    } 
    else 
    {
      // bias param already initialized
      bias_param_id_ = this->blobs_.size() - 1;
      bias_layer_->blobs()[0] = this->blobs_[bias_param_id_];
    }
    bias_propagate_down_.resize(1, false);
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}



template <typename Dtype>
void ScaleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const ScaleParameter& param = this->layer_param_.scale_param();
  Blob<Dtype>* scale = (bottom.size() > 1) ? bottom[1] : this->blobs_[0].get();
  // Always set axis_ == 0 in special case where scale is a scalar
  // (num_axes == 0). Mathematically equivalent for any choice of axis_, so the
  // actual setting can be safely ignored; and computation is most efficient
  // with axis_ == 0 and (therefore) outer_dim_ == 1. (Setting axis_ to
  // bottom[0]->num_axes() - 1, giving inner_dim_ == 1, would be equally
  // performant.)
  axis_ = (scale->num_axes() == 0) ?
      0 : bottom[0]->CanonicalAxisIndex(param.axis());
  CHECK_GE(bottom[0]->num_axes(), axis_ + scale->num_axes())
      << "scale blob's shape extends past bottom[0]'s shape when applied "
      << "starting with bottom[0] axis = " << axis_;
  for (int i = 0; i < scale->num_axes(); ++i) {
    CHECK_EQ(bottom[0]->shape(axis_ + i), scale->shape(i))
        << "dimension mismatch between bottom[0]->shape(" << axis_ + i
        << ") and scale->shape(" << i << ")";
  }
  outer_dim_ = bottom[0]->count(0, axis_);
  scale_dim_ = scale->count();
  inner_dim_ = bottom[0]->count(axis_ + scale->num_axes());
  if (bottom[0] == top[0]) {  // in-place computation
    temp_.ReshapeLike(*bottom[0]);
  } else {
    top[0]->ReshapeLike(*bottom[0]);
  }
  sum_result_.Reshape(vector<int>(1, outer_dim_ * scale_dim_));
  const int sum_mult_size = std::max(outer_dim_, inner_dim_);
  sum_multiplier_.Reshape(vector<int>(1, sum_mult_size));
  if (sum_multiplier_.cpu_data()[sum_mult_size - 1] != Dtype(1)) {
    caffe_set(sum_mult_size, Dtype(1), sum_multiplier_.mutable_cpu_data());
  }
  if (bias_layer_) {
    bias_bottom_vec_[0] = top[0];
    bias_layer_->Reshape(bias_bottom_vec_, top);
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  if (bottom[0] == top[0]) {
    //  In-place computation; need to store bottom data before overwriting it.
    //  Note that this is only necessary for Backward; we could skip this if not
    //  doing Backward, but Caffe currently provides no way of knowing whether
    //  we'll need to do Backward at the time of the Forward call.
    //  hhahahah  原来如此啊
    //  放到临时的空间里面。明白了。
    //   这里面又是完全浪费空间的哈
    caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(),
               temp_.mutable_cpu_data());
  }

  // Scale层主要完成 top=alpha∗bottom+betatop=alpha∗bottom+beta的过程，则层中主要有两个参数alphaalpha与betabeta,
  //求导会比较简单

  const Dtype* scale_data =
      ((bottom.size() > 1) ? bottom[1] : this->blobs_[0].get())->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  //
  for (int n = 0; n < outer_dim_; ++n)          //   N
  {
    for (int d = 0; d < scale_dim_; ++d)        //   C
    {
      //  每一个通道分别有一个对应的值。
      //  每一个通道一个值。
      //  每一个通道衬衣通道对应的参数值哈
      //  一个通道只有对应一个值
      //   所以参数的规模是C*1
      const Dtype factor = scale_data[d];      //  
      caffe_cpu_scale(inner_dim_, factor, bottom_data, top_data);
      bottom_data += inner_dim_;
      top_data += inner_dim_;
    }
  }
  if (bias_layer_) 
  {
    // 注意这里面调用了bias_layer_
    // 也就是每一个通道一个数值，不是乘以而是加上。
    // 这一层的top值自己输入的作为的bottom的值
    bias_layer_->Forward(bias_bottom_vec_, top);
  }
}

template <typename Dtype>
void ScaleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
  {
     if (bias_layer_ && this->param_propagate_down_[this->param_propagate_down_.size() - 1]) {
          bias_layer_->Backward(top, bias_propagate_down_, bias_bottom_vec_);
  }
  const bool scale_param = (bottom.size() == 1);
  LOG(INFO)<<"scale_param"<<scale_param;
  Blob<Dtype>* scale = scale_param ? this->blobs_[0].get() : bottom[1];
  if ((!scale_param && propagate_down[1]) ||(scale_param && this->param_propagate_down_[0])) 
  {
    const Dtype* top_diff = top[0]->cpu_diff();
    //   是不是原地操作指针是不是相等
    const bool in_place = (bottom[0] == top[0]);
    //   那么 存在temp 里面
    LOG(INFO)<<"in_place"<<in_place;
    const Dtype* bottom_data = (in_place ? &temp_ : bottom[0])->cpu_data();
    // Hack: store big eltwise product in bottom[0] diff, except in the special
    // case where this layer itself does the eltwise product, in which case we
    // can store it directly in the scale diff, and we're done.
    // If we're computing in-place (and not doing eltwise computation), this
    // hack doesn't work and we store the product in temp_.
    //LOG()
    const bool is_eltwise = (bottom[0]->count() == scale->count());
    LOG(INFO)<<"is_eltwise"<<is_eltwise;

      Dtype* product = (is_eltwise ? scale->mutable_cpu_diff() :
        (in_place ? temp_.mutable_cpu_data() : bottom[0]->mutable_cpu_diff())); 
          LOG(INFO)<<"inner_dim_"<<inner_dim_;
          LOG(INFO)<<"sum_result_.count()"<<sum_result_.count();
          LOG(INFO)<<"scale_param"<<scale_param;
          LOG(INFO)<<"outer_dim_"<<outer_dim_;
          LOG(INFO)<<"bottom.size()"<<bottom.size();
          LOG(INFO)<<"bottom[0]"<<bottom[1]->shape_string();
           //LOG(INFO<<""<<inner_dim_;
    caffe_mul(top[0]->count(), top_diff, bottom_data, product);
    if (!is_eltwise) 
    {
      Dtype* sum_result = NULL;
      if (inner_dim_ == 1) 
      {
        sum_result = product;
      } 
      else if (sum_result_.count() == 1) 
      {
        const Dtype* sum_mult = sum_multiplier_.cpu_data();
        Dtype* scale_diff = scale->mutable_cpu_diff();
        
        if (scale_param) 
        {
          Dtype result = caffe_cpu_dot(inner_dim_, product, sum_mult);
          *scale_diff += result;
        } else 
        {
          *scale_diff = caffe_cpu_dot(inner_dim_, product, sum_mult);
        }
      } 
      else 
      {
        const Dtype* sum_mult = sum_multiplier_.cpu_data();
        sum_result = (outer_dim_ == 1) ?
            scale->mutable_cpu_diff() : sum_result_.mutable_cpu_data();
        caffe_cpu_gemv(CblasNoTrans, sum_result_.count(), inner_dim_,
                       Dtype(1), product, sum_mult, Dtype(0), sum_result);
      }
      //  如果是多张图片
      if (outer_dim_ != 1) 
      {
        //   
        const Dtype* sum_mult = sum_multiplier_.cpu_data();
        //    A的梯度   
        Dtype* scale_diff = scale->mutable_cpu_diff();
        //    如果A是一个常数  
        if (scale_dim_ == 1) 
        {
          if (scale_param) 
          {
            Dtype result = caffe_cpu_dot(outer_dim_, sum_mult, sum_result);
            *scale_diff += result;
          } 
          else 
          {
            //  那么就是
            *scale_diff = caffe_cpu_dot(outer_dim_, sum_mult, sum_result);
          }
        } 
        else 
        {
          caffe_cpu_gemv(CblasTrans, outer_dim_, scale_dim_,
                         Dtype(1), sum_result, sum_mult, Dtype(scale_param),
                         scale_diff);
        }
      }
    }
   }
  //  并行的机会又来了
  if (propagate_down[0]) 
  {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* scale_data = scale->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int n = 0; n < outer_dim_; ++n) 
    {
      for (int d = 0; d < scale_dim_; ++d) 
      {
        const Dtype factor = scale_data[d];
        //   factor 是一个标量。
        //   也就是每一个通道的标量乘以对应的factor就是bottom的梯度
        caffe_cpu_scale(inner_dim_, factor, top_diff, bottom_diff);
        //     
        bottom_diff += inner_dim_;
        //    
        top_diff += inner_dim_;
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ScaleLayer);
#endif

INSTANTIATE_CLASS(ScaleLayer);
REGISTER_LAYER_CLASS(Scale);

}  // namespace caffe
