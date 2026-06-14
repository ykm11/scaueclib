#pragma once

#include <vector>
#include <pybind11/numpy.h>

class OnlineTVLA {
public:
    OnlineTVLA(size_t dim);

    void update(pybind11::array_t<float, 
            pybind11::array::c_style | pybind11::array::forcecast> new_data,
            int coin);
    
    pybind11::array_t<double> get_tvalue() const;

private:
    size_t num_g0_;
    size_t num_g1_;
    size_t dim_;

    std::vector<double> mean_g0_;
    std::vector<double> mean2_g0_;

    std::vector<double> mean_g1_;
    std::vector<double> mean2_g1_;

};

void bind_online_tvla(pybind11::module_& m);
