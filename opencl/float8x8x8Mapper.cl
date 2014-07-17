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
__kernel void float8x8x8Mapper(__global const float2 *T, __global float2 *TMapped)
{   
	float2 f;
	
    int gIdx0 = get_global_id(0);
    int gIdx1 = get_global_id(1);
    int gIdx2 = get_global_id(2);
    
    int I0 = get_global_size(0);
    int I1 = get_global_size(1);
    int I2 = get_global_size(2);
    
    int jumpI1 = I0;
    int jumpI2 = 2*I0*I1;
    
    //Calculate first index
    int idx = gIdx0 + 
        2 * gIdx1 * jumpI1 +
        2 * gIdx2 * jumpI2;

	jumpI2 -= 2*I0;

	int lIdx = get_local_id(0) + 4 * get_local_id(1) + 16 * get_local_id(2);
    int gIdx = get_group_id(0) + get_num_groups(0) * (get_group_id(1) + get_num_groups(1) * get_group_id(2));
    
    //Calculate first index
    int idx2 =  lIdx + 256 * gIdx;

    #pragma unroll
    for(int i1 = 0, j = 0; i1 < 2; i1++)
    {
        #pragma unroll
        for(int i2 = 0; i2 < 2; i2++, j++)
        {
            //Handle the 4 floats along the 0-axis
            f = T[idx];
            TMapped[idx2] = f;
			
            //Jump to next group along the 1-axis
            idx += jumpI1;
            idx2 += 64;
        }
        //Jump to next group along the 2-axis,
        //and undo jumps along the 1-axis
        idx += jumpI2;
    }
}
