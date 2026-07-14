#include <pybind11/pybind11.h>

#include "online_cpa.hpp"
#include "online_tvla.hpp"
#include "online_bivariate.hpp"
#include "online_wasserstein.hpp"

namespace py = pybind11;

PYBIND11_MODULE(scaueclib, m) {
    bind_online_cpa(m);
    bind_online_tvla(m);
    bind_online_bivariate(m);
    bind_online_wasserstein(m);
}
