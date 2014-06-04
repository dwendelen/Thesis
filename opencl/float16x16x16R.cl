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
__kernel void float16x16x16R(__global const float4 *T,
    __global const float4 *U0, __global const float4 *U1, __global const float4 *U2,
    int R, __global float *sum)
{
	
    __local float l[128];
    
    float4 a;
    float4 b[4];
    float4 c[16];
    float4 f;

    int I0 = get_global_size(0);
    int I1 = get_global_size(1);
    int I2 = get_global_size(2);
    
    #pragma unroll
    for(int i = 0; i < 16; i++)
    {
        c[i] = 0;
    }
    
    int gId0 = get_global_id(0);
    int gId1 = get_global_id(1);
    int gId2 = get_global_id(2);

    for(int r = 0; r < R; r++)
    {   
        //Fetch eerste 16 met n=0
        a = U0[gId0];

        //barrier(CLK_LOCAL_MEM_FENCE);
        
        //Fetch volgende 16 met n=1
        f = U1[gId1];
        
        b[0] = a * f.x;
        b[1] = a * f.y;
        b[2] = a * f.z;
        b[3] = a * f.w;

        //barrier(CLK_LOCAL_MEM_FENCE);
        
        //Fetch laatste 16 met n=2
        f = U2[gId2];
        
        #pragma unroll
        for(int i = 0; i < 4; i++)
        {
            c[i +  0] += b[i] * f.x;
            c[i +  4] += b[i] * f.y;
            c[i +  8] += b[i] * f.z;
            c[i + 12] += b[i] * f.w;
        }
        
        gId0 += I0;
        gId1 += I1;
        gId2 += I2;
    }
    //T-waarde aftrekken
    int gIdx0 = get_global_id(0);
    int gIdx1 = get_global_id(1);
    int gIdx2 = get_global_id(2);
    
    int lIdx = get_local_id(1) + 4 * get_local_id(2);
    int gIdx = get_group_id(0) + get_num_groups(0) * (get_group_id(1) + get_num_groups(1) * get_group_id(2));
    
    //Calculate first index
    int idx =  lIdx + 1024 * gIdx;   

	float4 s = (float4)(0.0f,0.0f,0.0f,0.0f);
	float4 t;

    #pragma unroll
    for(int i = 0; i < 16; i++)
    {
        //Handle the 4 floats along the 0-axis
        f = T[idx];
        t = c[i] - f;
		s += t*t;
			
        idx += 64;
    }

    bool bo = get_local_id(0) == 0 && 
              get_local_id(1) == 0 &&
              get_local_id(2) == 0;
    
    //By doing the index times two, every work-item uses another bank (2.411778 -> 2.343779)
    int index = 2*(get_local_id(0) + 4 * lIdx);
    l[index] = s.x + s.y + s.z + s.w;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(bo)
    {
        float sss = 0;
        #pragma unroll
        for(int i = 0; i < 2*64; i+=2)
        {
            sss += l[i];
        }
        
        sum[gIdx] = sss;
    }
}
