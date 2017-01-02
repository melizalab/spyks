#ifndef UTILITY_H
#define UTILITY_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace spyks {

template<typename T> inline
T const * interpolate(double t, T const * data, double dt, size_t NC)
{
        size_t index = std::round(t / dt);
        return data + index * NC;
}

template<typename T> inline
T square(T x) { return x * x; }

}

#endif
