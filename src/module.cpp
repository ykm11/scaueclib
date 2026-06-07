#include <pybind11/pybind11.h>

#include "online_cpa.hpp"

namespace py = pybind11;

PYBIND11_MODULE(scaueclib, m)
{
    bind_online_cpa(m);
}
