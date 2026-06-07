#include "online_cpa.hpp"

OnlineCPA::OnlineCPA(size_t dim) : 
    dim_(dim), num_(0), meanx_(dim), mean2x_(dim), meany_(0), mean2y_(0), covxy_(dim) {
}


pybind11::array_t<double> OnlineCPA::get_pearson_corr() const {
    pybind11::array_t<double> out(dim_);

    auto buf = out.request();
    double* x = static_cast<double*>(buf.ptr);

    const double* ptr_mean2x = mean2x_.data();
    const double* ptr_covxy = covxy_.data();

    const double inv_n = 1.0 / num_;
    double var_y = mean2y_ * inv_n;
    for (size_t i = 0; i < dim_; i++) { // out[i] = ...;
        double var_x = ptr_mean2x[i] * inv_n;
        double cov_xy = ptr_covxy[i] * inv_n;

        x[i] = cov_xy / std::sqrt(var_x * var_y);
    }

    return out;
}


void OnlineCPA::update(pybind11::array_t<float, 
        pybind11::array::c_style | pybind11::array::forcecast> array_x,
        double y) {

    num_++;
    const double inv_n = 1.0 / num_;

    auto buf = array_x.request();
    const float* x = static_cast<const float*>(buf.ptr);

    double* ptr_meanx = meanx_.data();
    double* ptr_mean2x = mean2x_.data();
    double* ptr_covxy = covxy_.data();

    // Update scalar Y
    double delta_y = y - meany_;
    meany_ += delta_y * inv_n;
    mean2y_ += delta_y * (y - meany_);

    // Update vector X and Cov[X, Y]
    for (size_t i = 0; i < dim_; i++) {
        double delta_x = (static_cast<double>(x[i]) - ptr_meanx[i]);
        ptr_meanx[i] += delta_x * inv_n;
        ptr_mean2x[i] += delta_x * (static_cast<double>(x[i]) - ptr_meanx[i]);

        ptr_covxy[i] += delta_x * (y - meany_); 
    }
    // divide N to obtain the actual (co)variace
}

void bind_online_cpa(pybind11::module_& m) {
    pybind11::class_<OnlineCPA>(m, "OnlineCPA")
        .def(pybind11::init<size_t>())
        .def("update", &OnlineCPA::update)
        .def("get_pearson_corr", &OnlineCPA::get_pearson_corr);
}
