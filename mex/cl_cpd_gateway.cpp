#include "mex.h"
#include <new>
#include <map>
#include <string>

//scalar, throwing new and it matching delete
void* operator new (std::size_t n) throw(std::bad_alloc)
{
    //mexPrintf("New");
    void* p = mxMalloc(n);
    if(p == NULL)
        throw std::bad_alloc();

    return p;
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

    }
