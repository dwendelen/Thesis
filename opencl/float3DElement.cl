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

__kernel void float3DElement(__global const float4 *T,
    __global const float4 *U0, __global const float4 *U1, __global const float4 *U2,
    int R, int I0, int I1, int I2,
    __global float *sum)
{   
	
    __local float l[128];
    float a;
    float f;
    float c;

    bool bo = get_global_id(0) == 0 && 
              get_global_id(1) == 0 &&
              get_global_id(2) == 0;
    
    c = 0;
    
    int gId0 = get_global_id(0);
    int gId1 = get_global_id(1);
    int gId2 = get_global_id(2);

    for(int r = 0; r < R; r++)
    {   
        a = U0[gId0];
        f = U1[gId1];
        a *= f;
        f = U2[gId2];
        a *= f
        
        gId0 += I0;
        gId1 += I1;
        gId2 += I2;
    }
    //T-waarde aftrekken
    int gIdx0 = get_global_id(0);
    int gIdx1 = get_global_id(1);
    int gIdx2 = get_global_id(2);
    
    int jumpI1 = I0;
    int jumpI2 = I0*I1;
    
    //Calculate index
    int idx = gIdx0 + 
        gIdx1 * jumpI1 +
        gIdx2 * jumpI2;

    f = T[idx];
    c -= f;
	c = c*c;

    if(bo)
    {   
        sum[0] = c;
    }
}
