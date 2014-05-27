

/*
I0  The number of elements along the 0-axis
I1  The number of elements along the 1-axis
I2  The number of elements along the 2-axis

U0  Expected shape: I0 x R
U1  Expected shape: I1 x R
U2  Expected shape: I2 x R
    In general: Expected serialization: idx = idxI + Ix*idxR

T   The 3D-tensor to approximate. Expected shape: I0 x I1 x I2
    Expected serialization: idx = idx0 + I0(idx1 + I1(idx2))
    
This kernel MUST be run with a local 4x4x4 workspace
*/
__attribute__((reqd_work_group_size(4, 4, 4)))
__kernel void float16x16x16E(__global const float4 *T,
    __global const float4 *U0, __global const float4 *U1, __global const float4 *U2,
    int R, int I0, int I1, int I2,
    __global float *sum)
{   
    
}
