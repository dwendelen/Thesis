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

IDEE Fetchen per float2 -> geen channel conflict
*/
__attribute__((reqd_work_group_size(4, 4, 4)))
__kernel void float8x8x8R(__global const float2 *T,
    __global const float2 *U0, __global const float2 *U1, __global const float2 *U2,
    int R, __global float *sum)
{
    __local float l[128];
    
    float2 subKr1;
    float2 subKr2[2];
    float2 localSum[4];
	float2 temp;

    int I0 = get_global_size(0);
    int I1 = get_global_size(1);
    int I2 = get_global_size(2);
    
    #pragma unroll
    for(int i = 0; i < 4; i++)
    {
        localSum[i] = 0;
    }
    
    int gId0 = get_global_id(0);
    int gId1 = get_global_id(1);
    int gId2 = get_global_id(2);

    for(int r = 0; r < R; r++)
    {   
        //Fetch eerste 4 met n=0
        subKr1 = U0[gId0];

        //barrier(CLK_LOCAL_MEM_FENCE);
        
        //Fetch volgende 4 met n=1
        temp = U1[gId1];
		subKr2[0] = subKr1 * temp.x;
		subKr2[1] = subKr1 * temp.y;

        //barrier(CLK_LOCAL_MEM_FENCE);
        
        //Fetch laatste 4 met n=2
        temp = U2[gId2];
        
        localSum[0] += subKr2[0] * temp.x;
        localSum[1] += subKr2[1] * temp.x;
        localSum[2] += subKr2[0] * temp.y;
        localSum[3] += subKr2[1] * temp.y;
        
        gId0 += I0;
        gId1 += I1;
        gId2 += I2;
    }
    //T-waarde aftrekken
    int gIdx0 = get_global_id(0);
    int gIdx1 = get_global_id(1);
    int gIdx2 = get_global_id(2);
    
    int lIdx = get_local_id(0) + 4 * get_local_id(1) + 16 * get_local_id(2);
    int gIdx = get_group_id(0) + get_num_groups(0) * (get_group_id(1) + get_num_groups(1) * get_group_id(2));
    
    //Calculate first index
    int idx =  lIdx + 256 * gIdx;   

    float2 s = 0;
	float2 t;

    #pragma unroll
    for(int i = 0; i < 4; i++)
    {
        //Handle the 4 floats along the 0-axis
        temp = T[idx];
        t = localSum[i] - temp;
		s += t*t;
			
        idx += 64;
    }

    bool bo = get_local_id(0) == 0 && 
              get_local_id(1) == 0 &&
              get_local_id(2) == 0;

	float sss = s.x + s.y;
    l[lIdx] = sss;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(bo)
    {        
        #pragma unroll
        for(int i = 1; i < 64; i++)
        {
            sss += l[i];
        }
        
        sum[gIdx] = sss;
    }
}
