#pragma OPENCL EXTENSION cl_amd_printf : enable

//The number of consecutive floats served by one channel
#define OFFSET      16

//LEN = OFFSET * nbWorkItems (=16)
#define LEN		  8192

//JUMP = LEN - OFFSET
#define JUMP       8176

/*
Basic idea:
To have two workitems per channel, to keep the channels busy.
The workitems MUST run on different CU's, to avoid stalling by other
workitems in the same wavefront.
*/
__attribute__((reqd_work_group_size(1,1,1)))
__kernel void Sum16(__global const float4 *array, const int n1, __global float *sum)
{ 	
    int idx = 0;
    int n = n1/4;
    float4 s = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for(idx = 0; idx < n; idx++)
    {
        s += array[idx];
    }
	
	
	sum[get_group_id(0)] = s.x + s.y + s.z + s.w;
}
