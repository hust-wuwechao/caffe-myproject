#include <algorithm>
#include <vector>

#include "caffe/layers/batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#define CUDNN_STREAMS_PER_GROUP 3
#define GROUP 1
template <typename Dtype>
void BatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
      use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
    eps_ = param.eps();
  if (this->blobs_.size() > 0) 
  {
    LOG(INFO) << "Skipping parameter initialization";
  } 
  else 
  {
    this->blobs_.resize(3);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));
    this->blobs_[1].reset(new Blob<Dtype>(sz));
    sz[0] = 1;
    this->blobs_[2].reset(new Blob<Dtype>(sz));
    for (int i = 0; i < 3; ++i) 
    {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
  // Mask statistics from optimization by setting local learning rates
  // for mean, variance, and the bias correction to zero.
  for (int i = 0; i < this->blobs_.size(); ++i) {
    if (this->layer_param_.param_size() == i) 
    {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      fixed_param_spec->set_lr_mult(0.f);
    } 
    else 
    {
      CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
          << "Cannot configure batch normalization statistics as layer "
          << "parameters.";
    }
  }
}



template <typename Dtype>
void BatchNormLayer<Dtype>::LayerSetUp1(
     const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top,
      cudnnHandle_t*  handle,
      cudaStream_t*   stream) 
{
 LOG(INFO)<<"进入了batchnor LayerSetUp1++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++";
  //  默认初始化3个handle。 其实这里面只需要一个
  //  所以我们改成了1.
  stream_=stream;
  handle_=new cublasHandle_t[GROUP*CUDNN_STREAMS_PER_GROUP];
  for(int i=0;i<3;i++)
  {
    cublasCreate(&handle_[i]);
    cublasSetStream(handle_[i],  stream_[i]);
  }
  BatchNormParameter param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  use_global_stats_ = this->phase_ == TEST;
  if (param.has_use_global_stats())
      use_global_stats_ = param.use_global_stats();
  if (bottom[0]->num_axes() == 1)
    channels_ = 1;
  else
    channels_ = bottom[0]->shape(1);
    eps_ = param.eps();
  if (this->blobs_.size() > 0) 
  {
    LOG(INFO) << "Skipping parameter initialization";
  } 
  else 
  {
    this->blobs_.resize(3);
    vector<int> sz;
    sz.push_back(channels_);
    this->blobs_[0].reset(new Blob<Dtype>(sz));
    this->blobs_[1].reset(new Blob<Dtype>(sz));
    sz[0] = 1;
    this->blobs_[2].reset(new Blob<Dtype>(sz));
    for (int i = 0; i < 3; ++i) 
    {
      caffe_set(this->blobs_[i]->count(), Dtype(0),
                this->blobs_[i]->mutable_cpu_data());
    }
  }
  // Mask statistics from optimization by setting local learning rates
  // for mean, variance, and the bias correction to zero.
  for (int i = 0; i < this->blobs_.size(); ++i) {
    if (this->layer_param_.param_size() == i) 
    {
      ParamSpec* fixed_param_spec = this->layer_param_.add_param();
      fixed_param_spec->set_lr_mult(0.f);
    } 
    else 
    {
      CHECK_EQ(this->layer_param_.param(i).lr_mult(), 0.f)
          << "Cannot configure batch normalization statistics as layer "
          << "parameters.";
    }
  }
}



template <typename Dtype>
void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (bottom[0]->num_axes() >= 1)
    CHECK_EQ(bottom[0]->shape(1), channels_);
  top[0]->ReshapeLike(*bottom[0]);

  vector<int> sz;
  sz.push_back(channels_);
  mean_.Reshape(sz);
  variance_.Reshape(sz);
  temp_.ReshapeLike(*bottom[0]);
  x_norm_.ReshapeLike(*bottom[0]);
  sz[0] = bottom[0]->shape(0);
  batch_sum_multiplier_.Reshape(sz);

  int spatial_dim = bottom[0]->count()/(channels_*bottom[0]->shape(0));
  if (spatial_sum_multiplier_.num_axes() == 0 ||
      spatial_sum_multiplier_.shape(0) != spatial_dim) {
    sz[0] = spatial_dim;
    spatial_sum_multiplier_.Reshape(sz);
    Dtype* multiplier_data = spatial_sum_multiplier_.mutable_cpu_data();
    caffe_set(spatial_sum_multiplier_.count(), Dtype(1), multiplier_data);
  }

  int numbychans = channels_*bottom[0]->shape(0);
  if (num_by_chans_.num_axes() == 0 ||
      num_by_chans_.shape(0) != numbychans) {
    sz[0] = numbychans;
    num_by_chans_.Reshape(sz);
    caffe_set(batch_sum_multiplier_.count(), Dtype(1),
        batch_sum_multiplier_.mutable_cpu_data());
  }
}

/* template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        //  N
        int num = bottom[0]->shape(0);
        //  H*W
        int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
        //
        if (bottom[0] != top[0]) 
        {
            caffe_copy(bottom[0]->count(), bottom_data, top_data);
        }

        if (use_global_stats_) 
        {
            // use the stored mean/variance estimates.
            const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
                0 : 1 / this->blobs_[2]->cpu_data()[0];
            caffe_cpu_scale(variance_.count(), scale_factor,
                this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
            caffe_cpu_scale(variance_.count(), scale_factor,
                this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
        } 
        else 
        {
            //  compute mean
            //  矩阵向量的乘法
            //  似的结果是  N*C个结果
            //  也就是求取了每一个通道的均值。
            caffe_cpu_gemv<Dtype>
            (
                CblasNoTrans,
                channels_ * num, 
                spatial_dim,
                1. / (num * spatial_dim),
                bottom_data,
                spatial_sum_multiplier_.cpu_data(),
                0.,
                num_by_chans_.mutable_cpu_data()
            );
            //  求取的多个样本的相同的通道的值进行求和， 得到C个数组。也就是跨通道的均值
            //  C*N 和 N*1  得到C*1的结果
            //  也就是不同样本的每一个通道的一个均值。 
            caffe_cpu_gemv<Dtype>
            (
                CblasTrans, 
                num, 
                channels_, 
                1.,
                num_by_chans_.cpu_data(),
                batch_sum_multiplier_.cpu_data(), 
                0.,
                mean_.mutable_cpu_data()
            );
        }

        // subtract mean
        //将得到的通道平均值扩张到N*C矩阵。
        caffe_cpu_gemm<Dtype>
        (
            CblasNoTrans, 
            CblasNoTrans, 
            num, 
            channels_, 
            1, 
            1,
            batch_sum_multiplier_.cpu_data(), 
            mean_.cpu_data(),
            0.,
            num_by_chans_.mutable_cpu_data()
        );

    // 由N*C*1 乘以 1*H*W  扩张到 （N*C）*（H*W）矩阵  乘以-1   和原来的值相加
    // 从而实现了每一个的数值的减去平均值。
    //  spatial_sum_multiplier_.cpu_data()是1*（H*W）是每一个的空间内部的通道的每一个值得权重
    caffe_cpu_gemm<Dtype>
    (
        CblasNoTrans, 
        CblasNoTrans,
        channels_ * num,
        spatial_dim, 
        1, 
        -1, 
        num_by_chans_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(),
        1.,
        top_data
    );

    if (!use_global_stats_) 
    {
        // compute variance using var(X) = E((X-EX)^2)
        //  对于每一个的差值去取平方
        caffe_sqr<Dtype>(top[0]->count(), top_data,temp_.mutable_cpu_data());  // (X-EX)^2
        //  得到每一个每一个样本每一个通道的误平方和。
        //  NC *1 
        caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
            1. / (num * spatial_dim), temp_.cpu_data(),
            spatial_sum_multiplier_.cpu_data(), 0.,
            num_by_chans_.mutable_cpu_data());
        // 求和不同样本的相同通道得方差。得到C*1，也就
        caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
            num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
            variance_.mutable_cpu_data());  // E((X_EX)^2)
        // compute and save moving average
        this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
        this->blobs_[2]->mutable_cpu_data()[0] += 1;
        //  blob_[0] = mean_ + moving_average_fraction_* blob_[0]; 
        //  把这个均值存储下来，对应于新的均值
        caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
            moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
        int m = bottom[0]->count()/channels_;
        Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
        //  对应于新的残差blob_[1] = bias_correction_factor * variance_ + moving_average_fraction_ * blob_[1]
        caffe_cpu_axpby(variance_.count(), bias_correction_factor,
            variance_.cpu_data(), moving_average_fraction_,
            this->blobs_[1]->mutable_cpu_data());
      }



        // normalize variance
        // 给梯度每一个元素加上一个常量，防止除数为0
        caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
        // 开方
        caffe_sqrt(variance_.count(), variance_.cpu_data(),
                    variance_.mutable_cpu_data());

        // replicate variance to input size
        // 方差扩张为NC
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
            batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
            num_by_chans_.mutable_cpu_data());
        //  方差扩展为NC*HW
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
            spatial_dim, 1, 1., num_by_chans_.cpu_data(),
            spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
        //   逐个元素相除
        caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
        // TODO(cdoersch): The caching is only needed because later in-place layers
        // might clobber the data.  Can we skip this if they won't?
        caffe_copy(x_norm_.count(), top_data,x_norm_.mutable_cpu_data());
} */

template <typename Dtype>
void BatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) 
{
        const Dtype* bottom_data = bottom[0]->cpu_data();
        Dtype* top_data = top[0]->mutable_cpu_data();
        //  N
        int num = bottom[0]->shape(0);
        //  H*W
        int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
        //  如果不是原地计算
        if (bottom[0] != top[0]) 
        {
            caffe_copy(bottom[0]->count(), bottom_data, top_data);
        }

        if (use_global_stats_) 
        {
            // use the stored mean/variance estimates.
            const Dtype scale_factor = this->blobs_[2]->cpu_data()[0] == 0 ?
                0 : 1 / this->blobs_[2]->cpu_data()[0];
            caffe_cpu_scale(variance_.count(), scale_factor,
                this->blobs_[0]->cpu_data(), mean_.mutable_cpu_data());
            caffe_cpu_scale(variance_.count(), scale_factor,
                this->blobs_[1]->cpu_data(), variance_.mutable_cpu_data());
        } 
        else 
        {
            //  compute mean
            //  矩阵向量的乘法
            //  似的结果是  N*C个结果
            //  也就是求取了每一个通道的均值。
            caffe_cpu_gemv<Dtype>
            (
                CblasNoTrans,
                channels_ * num, 
                spatial_dim,
                1. / (num * spatial_dim),
                bottom_data,
                spatial_sum_multiplier_.cpu_data(),
                0.,
                num_by_chans_.mutable_cpu_data()
            );
            //  求取的多个样本的相同的通道的值进行求和， 得到C个数组。也就是跨通道的均值
            //  C*N 和 N*1  得到C*1的结果
            //  也就是不同样本的每一个通道的一个均值。 
            caffe_cpu_gemv<Dtype>
            (
                CblasTrans, 
                num, 
                channels_, 
                1.,
                num_by_chans_.cpu_data(),
                batch_sum_multiplier_.cpu_data(), 
                0.,
                mean_.mutable_cpu_data()
            );
        }

        // subtract mean
        //将得到的通道平均值扩张到N*C矩阵。
        caffe_cpu_gemm<Dtype>
        (
            CblasNoTrans, 
            CblasNoTrans, 
            num, 
            channels_, 
            1, 
            1,
            batch_sum_multiplier_.cpu_data(), 
            mean_.cpu_data(),
            0.,
            num_by_chans_.mutable_cpu_data()
        );

    // 由N*C*1 乘以 1*H*W  扩张到 （N*C）*（H*W）矩阵  乘以-1   和原来的值相加
    // 从而实现了每一个的数值的减去平均值。
    //  spatial_sum_multiplier_.cpu_data()是1*（H*W）是每一个的空间内部的通道的每一个值得权重
    caffe_cpu_gemm<Dtype>
    (
        CblasNoTrans, 
        CblasNoTrans,
        channels_ * num,
        spatial_dim, 
        1, 
        -1, 
        num_by_chans_.cpu_data(),
        spatial_sum_multiplier_.cpu_data(),
        1.,
        top_data
    );

    if (!use_global_stats_) 
    {
        // compute variance using var(X) = E((X-EX)^2)
        //  对于每一个的差值去取平方

        caffe_sqr<Dtype>(top[0]->count(), top_data,temp_.mutable_cpu_data());  // (X-EX)^2
        //  得到每一个每一个样本每一个通道的误平方和。
        //  NC *1 
        caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim,
            1. / (num * spatial_dim), temp_.cpu_data(),
            spatial_sum_multiplier_.cpu_data(), 0.,
            num_by_chans_.mutable_cpu_data());
        // 求和不同样本的相同通道得方差。得到C*1，也就
        caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
            num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
            variance_.mutable_cpu_data());  // E((X_EX)^2)
        // compute and save moving average
        this->blobs_[2]->mutable_cpu_data()[0] *= moving_average_fraction_;
        this->blobs_[2]->mutable_cpu_data()[0] += 1;
        //  blob_[0] = mean_ + moving_average_fraction_* blob_[0]; 
        //  把这个均值存储下来，对应于新的均值
        caffe_cpu_axpby(mean_.count(), Dtype(1), mean_.cpu_data(),
            moving_average_fraction_, this->blobs_[0]->mutable_cpu_data());
        int m = bottom[0]->count()/channels_;
        Dtype bias_correction_factor = m > 1 ? Dtype(m)/(m-1) : 1;
        //  对应于新的残差blob_[1] = bias_correction_factor * variance_ + moving_average_fraction_ * blob_[1]
        caffe_cpu_axpby(variance_.count(), bias_correction_factor,
            variance_.cpu_data(), moving_average_fraction_,
            this->blobs_[1]->mutable_cpu_data());
      }



        //   normalize variance
        //   给梯度每一个元素加上一个常量，防止除数为0
        //   由于梯度本身是C*1
        caffe_add_scalar(variance_.count(), eps_, variance_.mutable_cpu_data());
        //    对于梯度进行开方
        //    
        caffe_sqrt(variance_.count(), variance_.cpu_data(),
                    variance_.mutable_cpu_data());

        //  replicate variance to input size
        //  方差由 c*1-扩张为NC
        //  
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
            batch_sum_multiplier_.cpu_data(), variance_.cpu_data(), 0.,
            num_by_chans_.mutable_cpu_data());
        //  方差扩展为NC*HW
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
            spatial_dim, 1, 1., num_by_chans_.cpu_data(),
            spatial_sum_multiplier_.cpu_data(), 0., temp_.mutable_cpu_data());
        //   逐个元素相除
        //  
        caffe_div(temp_.count(), top_data, temp_.cpu_data(), top_data);
        // TODO(cdoersch): The caching is only needed because later in-place layers
        // might clobber the data.  Can we skip this if they won't?
        caffe_copy(x_norm_.count(), top_data,x_norm_.mutable_cpu_data());
}










/* template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->cpu_diff();
  } else {
    caffe_copy(x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
    top_diff = x_norm_.cpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (use_global_stats_) {
    caffe_div(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
    return;
  }
  const Dtype* top_data = x_norm_.cpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());

  // reshape (broadcast) the above
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
}
 */

template <typename Dtype>
void BatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff;
  if (bottom[0] != top[0]) {
    top_diff = top[0]->cpu_diff();
  } else {
    caffe_copy(x_norm_.count(), top[0]->cpu_diff(), x_norm_.mutable_cpu_diff());
    top_diff = x_norm_.cpu_diff();
  }
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  if (use_global_stats_) {
    caffe_div(temp_.count(), top_diff, temp_.cpu_data(), bottom_diff);
    return;
  }
  const Dtype* top_data = x_norm_.cpu_data();
  int num = bottom[0]->shape()[0];
  int spatial_dim = bottom[0]->count()/(bottom[0]->shape(0)*channels_);
  // if Y = (X-mean(X))/(sqrt(var(X)+eps)), then
  //
  // dE(Y)/dX =
  //   (dE/dY - mean(dE/dY) - mean(dE/dY \cdot Y) \cdot Y)
  //     ./ sqrt(var(X) + eps)
  //
  // where \cdot and ./ are hadamard product and elementwise division,
  // respectively, dE/dY is the top diff, and mean/var/sum are all computed
  // along all dimensions except the channels dimension.  In the above
  // equation, the operations allow for expansion (i.e. broadcast) along all
  // dimensions except the channels dimension where required.

  // sum(dE/dY \cdot Y)
  caffe_mul(temp_.count(), top_data, top_diff, bottom_diff);
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      bottom_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());

  // reshape (broadcast) the above
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_ * num,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 0., bottom_diff);

  // sum(dE/dY \cdot Y) \cdot Y
  caffe_mul(temp_.count(), top_data, bottom_diff, bottom_diff);

  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemv<Dtype>(CblasNoTrans, channels_ * num, spatial_dim, 1.,
      top_diff, spatial_sum_multiplier_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemv<Dtype>(CblasTrans, num, channels_, 1.,
      num_by_chans_.cpu_data(), batch_sum_multiplier_.cpu_data(), 0.,
      mean_.mutable_cpu_data());
  // reshape (broadcast) the above to make
  // sum(dE/dY)-sum(dE/dY \cdot Y) \cdot Y
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels_, 1, 1,
      batch_sum_multiplier_.cpu_data(), mean_.cpu_data(), 0.,
      num_by_chans_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num * channels_,
      spatial_dim, 1, 1., num_by_chans_.cpu_data(),
      spatial_sum_multiplier_.cpu_data(), 1., bottom_diff);

  // dE/dY - mean(dE/dY)-mean(dE/dY \cdot Y) \cdot Y
  caffe_cpu_axpby(temp_.count(), Dtype(1), top_diff,
      Dtype(-1. / (num * spatial_dim)), bottom_diff);

  // note: temp_ still contains sqrt(var(X)+eps), computed during the forward
  // pass.
  caffe_div(temp_.count(), bottom_diff, temp_.cpu_data(), bottom_diff);
}

#ifdef CPU_ONLY
STUB_GPU(BatchNormLayer);
#endif

INSTANTIATE_CLASS(BatchNormLayer);
REGISTER_LAYER_CLASS(BatchNorm);
}  // namespace caffe
