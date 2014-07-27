#pragma OPENCL EXTENSION cl_amd_printf : enable

/*
Mogelijk issue: Channel conflict omdat er gesprongen wordt
met een veelvoud van 4x16=64 bytes en een veelvoud van
64x4 = 256 bytes voor T.
*/

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
__kernel void Kernel(__global const float4 *T,
    __global const float4 *U0, __global const float4 *U1, __global const float4 *U2,
    int R, __global float *sum)
{   
	
    __local float l[128];
    
    /* */
    
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
    
    int jumpI1 = I0;
    int jumpI2 = 4*I0*I1;
    
    //Calculate first index
    int idx = gIdx0 + 
        4*gIdx1 * jumpI1 +
        4*gIdx2 * jumpI2;

	jumpI2 -= 4*I0;

	float4 s = (float4)(0.0f,0.0f,0.0f,0.0f);
	float4 t;

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
        idx += jumpI2;
    }

    
    
    //By doing the index times two, every work-item uses another bank (2.411778 -> 2.343779) 
    int index = 2*(get_local_id(0) + 4 * get_local_id(1) + 16 * get_local_id(2));
    l[index] = s.x + s.y + s.z + s.w;
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    bool isFirstWorkItem = get_local_id(0) == 0 &&
              get_local_id(1) == 0 &&
              get_local_id(2) == 0;
    
    if(isFirstWorkItem)
    {
        float sss = 0;
        #pragma unroll
        for(int i = 0; i < 2*64; i+=2)
        {
            sss += l[i];
        }
        
        index = get_group_id(0) + get_num_groups(0) * (get_group_id(1) + get_num_groups(1) * get_group_id(2));
        sum[index] = sss;
    }
}

/*
Voor:
R = 4
I = 360 -> 368 elementen

Enkel begin:
Adressen 0,1,3,4,5, 7,8



Workitem (0 0 0)
Initial byte address: 0
That is channel: 0

Workitem (1 0 0)
Initial byte address: 4
That is channel: 0

Workitem (2 0 0)
Initial byte address: 8
That is channel: 0

Workitem (3 0 0)
Initial byte address: 12
That is channel: 0

Workitem (0 1 0)
Initial byte address: 1472
That is channel: 5

Workitem (1 1 0)
Initial byte address: 1476
That is channel: 5

Workitem (2 1 0)
Initial byte address: 1480
That is channel: 5

Workitem (3 1 0)
Initial byte address: 1484
That is channel: 5

Workitem (0 2 0)
Initial byte address: 2944
That is channel: 3

Workitem (1 2 0)
Initial byte address: 2948
That is channel: 3

Workitem (2 2 0)
Initial byte address: 2952
That is channel: 3

Workitem (3 2 0)
Initial byte address: 2956
That is channel: 3

Workitem (0 3 0)
Initial byte address: 4416
That is channel: 1

Workitem (1 3 0)
Initial byte address: 4420
That is channel: 1

Workitem (2 3 0)
Initial byte address: 4424
That is channel: 1

Workitem (3 3 0)
Initial byte address: 4428
That is channel: 1

Workitem (0 0 1)
Initial byte address: 541696
That is channel: 4

Workitem (1 0 1)
Initial byte address: 541700
That is channel: 4

Workitem (2 0 1)
Initial byte address: 541704
That is channel: 4

Workitem (3 0 1)
Initial byte address: 541708
That is channel: 4

Workitem (0 1 1)
Initial byte address: 543168
That is channel: 1

Workitem (1 1 1)
Initial byte address: 543172
That is channel: 1

Workitem (2 1 1)
Initial byte address: 543176
That is channel: 1

Workitem (3 1 1)
Initial byte address: 543180
That is channel: 1

Workitem (0 2 1)
Initial byte address: 544640
That is channel: 7

Workitem (1 2 1)
Initial byte address: 544644
That is channel: 7

Workitem (2 2 1)
Initial byte address: 544648
That is channel: 7

Workitem (3 2 1)
Initial byte address: 544652
That is channel: 7

Workitem (0 3 1)
Initial byte address: 546112
That is channel: 5

Workitem (1 3 1)
Initial byte address: 546116
That is channel: 5

Workitem (2 3 1)
Initial byte address: 546120
That is channel: 5

Workitem (3 3 1)
Initial byte address: 546124
That is channel: 5

Workitem (0 0 2)
Initial byte address: 1083392
That is channel: 0

Workitem (1 0 2)
Initial byte address: 1083396
That is channel: 0

Workitem (2 0 2)
Initial byte address: 1083400
That is channel: 0

Workitem (3 0 2)
Initial byte address: 1083404
That is channel: 0

Workitem (0 1 2)
Initial byte address: 1084864
That is channel: 5

Workitem (1 1 2)
Initial byte address: 1084868
That is channel: 5

Workitem (2 1 2)
Initial byte address: 1084872
That is channel: 5

Workitem (3 1 2)
Initial byte address: 1084876
That is channel: 5

Workitem (0 2 2)
Initial byte address: 1086336
That is channel: 3

Workitem (1 2 2)
Initial byte address: 1086340
That is channel: 3

Workitem (2 2 2)
Initial byte address: 1086344
That is channel: 3

Workitem (3 2 2)
Initial byte address: 1086348
That is channel: 3

Workitem (0 3 2)
Initial byte address: 1087808
That is channel: 1

Workitem (1 3 2)
Initial byte address: 1087812
That is channel: 1

Workitem (2 3 2)
Initial byte address: 1087816
That is channel: 1

Workitem (3 3 2)
Initial byte address: 1087820
That is channel: 1

Workitem (0 0 3)
Initial byte address: 1625088
That is channel: 4

Workitem (1 0 3)
Initial byte address: 1625092
That is channel: 4

Workitem (2 0 3)
Initial byte address: 1625096
That is channel: 4

Workitem (3 0 3)
Initial byte address: 1625100
That is channel: 4

Workitem (0 1 3)
Initial byte address: 1626560
That is channel: 1

Workitem (1 1 3)
Initial byte address: 1626564
That is channel: 1

Workitem (2 1 3)
Initial byte address: 1626568
That is channel: 1

Workitem (3 1 3)
Initial byte address: 1626572
That is channel: 1

Workitem (0 2 3)
Initial byte address: 1628032
That is channel: 7

Workitem (1 2 3)
Initial byte address: 1628036
That is channel: 7

Workitem (2 2 3)
Initial byte address: 1628040
That is channel: 7

Workitem (3 2 3)
Initial byte address: 1628044
That is channel: 7

Workitem (0 3 3)
Initial byte address: 1629504
That is channel: 5

Workitem (1 3 3)
Initial byte address: 1629508
That is channel: 5
*/
