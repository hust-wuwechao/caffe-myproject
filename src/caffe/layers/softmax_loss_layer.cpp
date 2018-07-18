#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

#define CUDNN_STREAMS_PER_GROUP 3
#define GROUP 1



template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);

  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}


template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp1(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top,
    cudnnHandle_t* handle, 
    cudaStream_t*  stream) 
  {
 
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  stream_=stream;
  handle_=new cublasHandle_t[GROUP*CUDNN_STREAMS_PER_GROUP];
  for(int i=0;i<3;i++)
  {
    cublasCreate(&handle_[i]);
    cublasSetStream(handle_[i],  stream_[i]);
  } 
  /* stream_=stream;
  handle_=handle; */
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  
  LOG(INFO)<<"在这里进行调用构造softmax层";
  LOG_IF(INFO, Caffe::root_solver())
        << "typeid(x).name() "<<typeid(&softmax_layer_).name();

  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_,handle,stream);
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_)
 {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } 
  else 
  {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}




template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  //  
  const Dtype* prob_data = prob_.cpu_data();

  const Dtype* label = bottom[1]->cpu_data();

  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  //  首先认为所有的损失都是0
  //  batch 大小
  for (int i = 0; i < outer_num_; ++i) 
  {
    //  =1
    for (int j = 0; j < inner_num_; j++) 
    {
      //  
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) 
      {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      //  损失等于+
      //  样本原本的标签的值对应的概率的然后选择取 log ()
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }
   //  没有被呼略有效样本的个数。
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);

  if (top.size() == 2) 
  {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) 
{
  if (propagate_down[1])
  {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) 
  {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    //  首先，bottom  diff的值就是softmax  的输出值。

    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i)
     {
      for (int j = 0; j < inner_num_; ++j) 
      {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        //  对应的标签被忽略
        if (has_ignore_label_ && label_value == ignore_label_)
        {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c)
          {
            //  他的梯度都设置0
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        }
        else 
        {
          //  对应的标签位置的梯度的值会减一
          //  其他位置的梯度为保持不变。
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
