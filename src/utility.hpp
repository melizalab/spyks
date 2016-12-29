#ifndef UTILITY_H
#define UTILITY_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace spyks {

template<typename T>
T const * interpolate(double t, T const * data, double dt, size_t NC)
{
        size_t index = std::round(t / dt);
        return data + index * NC;
}

/** an observer that writes to a numpy array */
template <typename Model>
struct pyarray_writer {
        typedef typename Model::state_type state_type;
        pyarray_writer(size_t nsteps)
                : step(0), X(py::dtype::of<double>(), {nsteps, Model::N_STATE}) {}
        void operator()(state_type const & x, double time) {
                double * dptr = static_cast<double*>(X.mutable_data(step));
                std::copy_n(x.begin(), Model::N_STATE, dptr);
                ++step;
        }
        size_t step;
        py::array X;
};


}

#endif
