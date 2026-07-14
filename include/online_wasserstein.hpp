#pragma once

#include <vector>
#include <pybind11/numpy.h>

class OnlineWasserstein {
public:
    OnlineWasserstein(size_t dim, uint8_t adc_bit);

    void update(pybind11::array_t<float, 
            pybind11::array::c_style | pybind11::array::forcecast> new_data,
            int coin);
    
    pybind11::array_t<double> get_distance() const;
    //void clear();

private:
    size_t num_g0_;
    size_t num_g1_;
    size_t dim_;
    uint16_t binsize_;

    //std::vector<double> cumsum_0_;
    //std::vector<double> cumsum_1_;

    std::vector<uint32_t> histo_0_;
    std::vector<uint32_t> histo_1_;
};

void bind_online_wasserstein(pybind11::module_& m);
