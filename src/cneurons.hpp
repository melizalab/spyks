#ifndef _SPYKES_H
#define _SPYKES_H

#include <vector>

typedef double dtype;
typedef std::vector<dtype> vtype;

class neuron 
{
public:
    virtual ~neuron();

    virtual void set(theta) = 0;
    virtual vtype get(theta) = 0;
    virtual void run(stim) = 0;
};

#endif
