#include <algorithm>
#include <vector>
#include "xmmintrin.h"

#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template<typename T, typename U>
struct is_same
{
    static const bool value = false;
};

template<typename T>
struct is_same<T, T>
{
    static const bool value = true;
};

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  if (this->layer_param_.relu_param().has_negative_slope()) {
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0))
          + negative_slope * std::min(bottom_data[i], Dtype(0));
    }
  } else {
#ifdef USE_ALIGNED_MALLOC
    if (is_same<Dtype, float>::value) {
      __m128 *bottom_sse = (__m128*)bottom_data;
      __m128 *top_sse = (__m128*)top_data;
      __m128 zero = _mm_setzero_ps();
      for (int i = 0; i < (count + 3)/4; i++) {
        top_sse[i] = _mm_max_ps(bottom_sse[i], zero);
      }
    } else {
      for (int i = 0; i < count; ++i) {
        top_data[i] = std::max(bottom_data[i], Dtype(0));
      }
    }
#else
    for (int i = 0; i < count; ++i) {
      top_data[i] = std::max(bottom_data[i], Dtype(0));
    }
#endif
  }
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLULayer);
#endif

INSTANTIATE_CLASS(ReLULayer);

}  // namespace caffe
