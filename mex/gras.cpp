#include "mex.h"
#include <new>
#include <map>
#include <string>

#include "echo.hpp"

//scalar, throwing new and it matching delete
void* operator new (std::size_t n) throw(std::bad_alloc)
{
    //mexPrintf("New");
    void* p = mxMalloc(n);
    if(p == NULL)
        throw std::bad_alloc();
}
void operator delete (void* p) throw()
{
    //mexPrintf("Del");
    mxFree(p);
}

//scalar, nothrow new and it matching delete
void* operator new (std::size_t n,const std::nothrow_t&) throw()
{
    //mexPrintf("New");
    return mxMalloc(n);
} 
void operator delete (void* p, const std::nothrow_t&) throw()
{
    //mexPrintf("Del");
    mxFree(p);
}


void mexFunction(int nlhs, mxArray *plhs[],
    int nrhs, const mxArray *prhs[])
    {
        std::map<std::string, int> *m = new std::map<std::string, int>();
        (*m)["Test"] = 5;
        delete m;
    }
