#include "online_tvla.hpp"

OnlineTVLA::OnlineTVLA(size_t dim) : 
    dim_(dim), num_g0_(0), num_g1_(0),
    mean_g0_(dim), m2_g0_(dim), mean_g1_(dim), m2_g1_(dim) {
}


pybind11::array_t<double> OnlineTVLA::get_tvalue() const {
    pybind11::array_t<double> out(dim_);

    auto buf = out.request();
    double* x = static_cast<double*>(buf.ptr);

    const double* ptr_mean_g0 = mean_g0_.data();
    const double* ptr_mean_g1 = mean_g1_.data();

    const double* ptr_m2_g0 = m2_g0_.data();
    const double* ptr_m2_g1 = m2_g1_.data();

    const double inv_n_g0 = 1.0 / num_g0_;
    const double inv_n_g1 = 1.0 / num_g1_;

    for (size_t i = 0; i < dim_; i++) { // out[i] = ...;
        double var_g0 = ptr_m2_g0[i] * inv_n_g0;
        double var_g1 = ptr_m2_g1[i] * inv_n_g1;

        // t = (m0 - m1) / sqrt ( var_g0/num_g0 + var_g1/num_g1 )
        x[i] = (ptr_mean_g0[i] - ptr_mean_g1[i]) \
               / std::sqrt( var_g0*inv_n_g0 + var_g1*inv_n_g1 );
    }

    return out;
}


void OnlineTVLA::update(pybind11::array_t<float, 
        pybind11::array::c_style | pybind11::array::forcecast> array_x,
        int coin) {

    auto buf = array_x.request();
    const float* x = static_cast<const float*>(buf.ptr);
    double* ptr_mean;
    double* ptr_m2;


    double inv_n;
    if (coin == 0) {
      num_g0_++;
      inv_n = 1.0 / num_g0_;
      ptr_mean = mean_g0_.data();
      ptr_m2 = m2_g0_.data();

    } else if (coin == 1) {
      num_g1_++;
      inv_n = 1.0 / num_g1_;
      ptr_mean = mean_g1_.data();
      ptr_m2 = m2_g1_.data();

    } else {

    }

    // Update vector X 
    for (size_t i = 0; i < dim_; i++) {
        double delta_x = (static_cast<double>(x[i]) - ptr_mean[i]);
        ptr_mean[i] += delta_x * inv_n;
        ptr_m2[i] += delta_x * (static_cast<double>(x[i]) - ptr_mean[i]);

    }
    // divide N to obtain the actual variace
}

void OnlineTVLA::clear() {
  std::fill(mean_g0_.begin(), mean_g0_.end(), 0);
  std::fill(mean_g1_.begin(), mean_g1_.end(), 0);
  std::fill(m2_g0_.begin(), m2_g0_.end(), 0);
  std::fill(m2_g1_.begin(), m2_g1_.end(), 0);

  num_g0_ = 0;
  num_g1_ = 0;
}

void bind_online_tvla(pybind11::module_& m) {
    pybind11::class_<OnlineTVLA>(m, "OnlineTVLA")
        .def(pybind11::init<size_t>())
        .def("update", &OnlineTVLA::update)
        .def("clear", &OnlineTVLA::clear)
        .def("get_tvalue", &OnlineTVLA::get_tvalue);
}
