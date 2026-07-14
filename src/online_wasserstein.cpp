#include "online_wasserstein.hpp"

OnlineWasserstein::OnlineWasserstein(size_t dim, uint8_t adc_bit) : 
    dim_(dim), num_g0_(0), num_g1_(0), binsize_(1ULL << adc_bit),
    histo_0_(dim * (1ULL << adc_bit)),
    histo_1_(dim * (1ULL << adc_bit)) {
}


pybind11::array_t<double> OnlineWasserstein::get_distance() const {
    pybind11::array_t<double> out(dim_);

    auto buf = out.request();
    double* x = static_cast<double*>(buf.ptr);

    const double inv_n_g0 = 1.0 / num_g0_;
    const double inv_n_g1 = 1.0 / num_g1_;

    double tmp_cumsum0, tmp_cumsum1;
    double distance;

    const uint32_t* ptr_histo_0 = histo_0_.data();
    const uint32_t* ptr_histo_1 = histo_1_.data();

    for (size_t i = 0; i < dim_; i++) { // out[i] = ...;

      tmp_cumsum0 = 0.0;
      tmp_cumsum1 = 0.0;
      distance = 0.0;
      for (size_t j = 0; j < binsize_; j++) {
        tmp_cumsum0 += histo_0_[i * binsize_ + j] * inv_n_g0;
        tmp_cumsum1 += histo_1_[i * binsize_ + j] * inv_n_g1;
        distance += std::abs(tmp_cumsum0 - tmp_cumsum1);
      }
      x[i] = distance;
    }

    return out;
}


void OnlineWasserstein::update(pybind11::array_t<float, 
        pybind11::array::c_style | pybind11::array::forcecast> array_x,
        int coin) {

    auto buf = array_x.request();
    const float* x = static_cast<const float*>(buf.ptr);
    uint32_t* ptr_histo;

    if (coin == 0) {
      num_g0_++;
      ptr_histo = histo_0_.data();

    } else if (coin == 1) {
      num_g1_++;
      ptr_histo = histo_1_.data();

    } else {

    }

    // Update the histogram
    for (size_t i = 0; i < dim_; i++) {
        uint32_t idx = static_cast<uint32_t>(x[i]);
        ptr_histo[i * binsize_ + idx] += 1;

    }
}

/*
void OnlineWasserstein::clear() {

  num_g0_ = 0;
  num_g1_ = 0;
}
*/

void bind_online_wasserstein(pybind11::module_& m) {
    pybind11::class_<OnlineWasserstein>(m, "OnlineWasserstein")
        .def(pybind11::init<size_t, uint8_t>())
        .def("update", &OnlineWasserstein::update)
        //.def("clear", &OnlineTVLA::clear)
        .def("get_distance", &OnlineWasserstein::get_distance);
}
