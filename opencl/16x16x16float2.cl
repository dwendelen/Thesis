#pragma OPENCL EXTENSION cl_amd_printf : enable

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
__kernel void float16x16x16(__global const float4 *T,
    __global const float4 *U0, __global const float4 *U1, __global const float4 *U2,
    __local float4 *l, int R, int I0, int I1, int I2,
    __global float *sum)
{   
    float4 a;
    float4 b[4];
    float4 c[16];
    float4 f;

    float4 s = (float4)(0.0f,0.0f,0.0f,0.0f);
    float4 t;

    bool bo = get_global_id(0) == 0  && 
              get_global_id(1) == 0 &&
              get_global_id(2) == 0;
    
    int iterations0 = I0/4;
    int iterations1 = I1/4;
    int iterations2 = I2/4;
    
    //printf("Its %i", iterations0);
    
    int gIdx0 = get_global_id(0);// * iterations0;
    int gIdx1 = get_global_id(1);// * iterations1;
    int gIdx2 = get_global_id(2);// * iterations2;
    
    int jumpI1 = I0;
    int jumpI2 = 4*I0*I1;
    
    //for(int e2 = 0; e2 < iterations2; e2++)
    //{
        //for(int e1 = 0; e1 < iterations1; e1++)
        //{
            //for(int e0 = 0; e0 < iterations0; e0++)
            //{
                #pragma unroll
                for(int i = 0; i < 16; i++)
                {
                    c[i] = 0;
                }
                
                int gId0 = gIdx0;
                int gId1 = gIdx1;
                int gId2 = gIdx2;

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
                
                //Calculate first index
                int idx = gIdx0 + 
                    4*gIdx1 * jumpI1 +
                    4*gIdx2 * jumpI2;

                #pragma unroll
                for(int i1 = 0, j = 0; i1 < 4; i1++)
                {
                    #pragma unroll
                    for(int i2 = 0; i2 < 4; i2++, j++)
                    {
                        //Handle the 4 floats along the 0-axis
                        f = T[idx];
                        t = c[j] - f;
			            s += t*t;
			
                        //Jump to next group along the 1-axis
                        idx += jumpI1;
                    }
                    //Jump to next group along the 2-axis,
                    //and undo jumps along the 1-axis
                    idx += jumpI2 - 4*jumpI1;
                }
                
                //gIdx0++;
            //}
            //gIdx1++;
        //}
        //gIdx2++;
    //}

    if(bo)
    {   
        sum[0] = s.x + s.y + s.z + s.w;
    }
}
