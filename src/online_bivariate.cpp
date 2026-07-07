#include "online_bivariate.hpp"

OnlineBivariate::OnlineBivariate(size_t dim) : 
    dim_(dim), num_(0), old_delta_(dim_), new_delta_(dim_), mean_(dim), 
    cov_(dim, std::vector<double>(dim)),
    z_mean_(dim, std::vector<double>(dim)), z_M2_(dim, std::vector<double>(dim))
    {
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

    // Update means
    for (size_t i = 0; i < dim_; i++) {
        double xi = static_cast<double>(x[i]);
        old_delta_[i] = xi - mean_[i];
        mean_[i] += old_delta_[i] * inv_n;
        new_delta_[i] = xi - mean_[i];
    }

    // Update covariaces
    for (size_t i = 0; i < dim_; i++) {
        for (size_t j = 0; j < dim_; j++) {
            double z_ij = old_delta_[i] * new_delta_[j];
            cov_[i][j] += z_ij;

            // for t-test
            double z_delta = z_ij - z_mean_[i][j];
            z_mean_[i][j] += z_delta * inv_n;
            z_M2_[i][j]   += z_delta * (z_ij - z_mean_[i][j]);
        }
    }
    // divide N to obtain the actual covariace
}

pybind11::array_t<double>
OnlineBivariate::welch_t_against(const OnlineBivariate& other) const {

    pybind11::array_t<double> out({dim_, dim_});

    auto buf = out.request();
    double* t = static_cast<double*>(buf.ptr);

    const double inv_n0 = 1.0 / static_cast<double>(num_);
    const double inv_n1 = 1.0 / static_cast<double>(other.num_);

    for (size_t i = 0; i < dim_; i++) {
        for (size_t j = 0; j < dim_; j++) {

            const double mean0 = z_mean_[i][j];
            const double mean1 = other.z_mean_[i][j];

            const double var0 = z_M2_[i][j] * inv_n0;
            const double var1 = other.z_M2_[i][j] * inv_n1;

            const double denom =
                std::sqrt(var0 * inv_n0 + var1 * inv_n1);

            t[i * dim_ + j] = (mean0 - mean1) / denom;
        }
    }

    return out;
}

/*
void OnlineBivariate::clear() {
  std::fill(mean_.begin(), mean_.end(), 0);

  num_ = 0;
}
*/


OnlineBivariateTVLA::OnlineBivariateTVLA(size_t dim) : 
    g0_(dim), g1_(dim)
    {
}


void OnlineBivariateTVLA::update(pybind11::array_t<float, 
        pybind11::array::c_style | pybind11::array::forcecast> array_x,
        int coin) {

    if (coin == 0) {
        g0_.update(array_x);
    } else if (coin == 1) {
        g1_.update(array_x);
    } else{

    }

}

pybind11::array_t<double>
OnlineBivariateTVLA::get_tvalue() const {
    pybind11::array_t<double> out({dim_, dim_});

    out = g0_.welch_t_against(g1_);
    return out;
}



void bind_online_bivariate(pybind11::module_& m) {
    pybind11::class_<OnlineBivariate>(m, "OnlineBivariate")
        .def(pybind11::init<size_t>())
        .def("update", &OnlineBivariate::update)
        .def("welch_t_against", &OnlineBivariate::welch_t_against)
        //.def("clear", &OnlineBivariate::clear)
        .def("get_cov", &OnlineBivariate::get_cov);

    pybind11::class_<OnlineBivariateTVLA>(m, "OnlineBivariateTVLA")
        .def(pybind11::init<size_t>())
        .def("update", &OnlineBivariateTVLA::update)
        .def("get_tvalue", &OnlineBivariateTVLA::get_tvalue);
}
