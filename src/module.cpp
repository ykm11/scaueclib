#include <pybind11/pybind11.h>

#include "online_cpa.hpp"
#include "online_tvla.hpp"

namespace py = pybind11;

PYBIND11_MODULE(scaueclib, m) {
    bind_online_cpa(m);
    bind_online_tvla(m);
}
