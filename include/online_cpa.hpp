#pragma once

#include <vector>
#include <pybind11/numpy.h>

class OnlineCPA {
public:
    OnlineCPA(size_t dim);

    void update(pybind11::array_t<float, 
            pybind11::array::c_style | pybind11::array::forcecast> new_data,
            double label);
    
    pybind11::array_t<double> get_pearson_corr() const;

private:
    size_t num_;
    size_t dim_;

    std::vector<double> meanx_;
    std::vector<double> mean2x_;
    std::vector<double> covxy_;

    double meany_, mean2y_;
};

void bind_online_cpa(pybind11::module_& m);
