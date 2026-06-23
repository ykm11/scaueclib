#pragma once

#include <vector>
#include <pybind11/numpy.h>

class OnlineBivariate {
public:
    OnlineBivariate(size_t dim);

    void update(pybind11::array_t<float, 
            pybind11::array::c_style | pybind11::array::forcecast> new_data
            );
    
    pybind11::array_t<double> get_cov() const;
    //void clear();

private:
    size_t num_;
    size_t dim_;

    std::vector<double> old_delta_;
    std::vector<double> new_delta_;
    std::vector<double> mean_;
    std::vector< std::vector<double> > cov_;

};

void bind_online_bivariate(pybind11::module_& m);
