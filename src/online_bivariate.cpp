#include "online_bivariate.hpp"

OnlineBivariate::OnlineBivariate(size_t dim) : 
    dim_(dim), num_(0), old_delta_(dim_), new_delta_(dim_),
    mean_(dim), cov_(dim, std::vector<double>(dim)) {
}


pybind11::array_t<double> OnlineBivariate::get_cov() const {
    pybind11::array_t<double> out({dim_, dim_});

    auto buf = out.request();
    double* x = static_cast<double*>(buf.ptr);
    const double inv_n = 1.0 / num_;

    for (size_t i = 0; i < dim_; i++) {
        for (size_t j = 0; j < dim_; j++) {
            x[i * dim_ + j] = cov_[i][j] * inv_n;
        }
    }

    return out;
}


void OnlineBivariate::update(pybind11::array_t<float, 
        pybind11::array::c_style | pybind11::array::forcecast> array_x
        ) {

    auto buf = array_x.request();
    const float* x = static_cast<const float*>(buf.ptr);

    double* ptr_mean = mean_.data();

    num_++;
    double inv_n = 1.0 / num_;

    // Update vector
    for (size_t i = 0; i < dim_; i++) {
        double xi = static_cast<double>(x[i]);
        old_delta_[i] = xi - mean_[i];
        mean_[i] += old_delta_[i] * inv_n;
        new_delta_[i] = xi - mean_[i];
    }

    for (size_t i = 0; i < dim_; i++) {
        for (size_t j = 0; j < dim_; j++) {
            cov_[i][j] += old_delta_[i] * new_delta_[j];
        }
    }
    // divide N to obtain the actual covariace
}

/*
void OnlineBivariate::clear() {
  std::fill(mean_.begin(), mean_.end(), 0);

  num_ = 0;
}
*/

void bind_online_bivariate(pybind11::module_& m) {
    pybind11::class_<OnlineBivariate>(m, "OnlineBivariate")
        .def(pybind11::init<size_t>())
        .def("update", &OnlineBivariate::update)
        //.def("clear", &OnlineBivariate::clear)
        .def("get_cov", &OnlineBivariate::get_cov);
}
