#ifndef UTILITY_H
#define UTILITY_H

namespace spyks {

template<typename T>
T const * interpolate(double t, T const * data, double dt, size_t NC)
{
        size_t index = std::round(t / dt);
        return data + index * NC;
}

}

#endif
