#pragma OPENCL EXTENSION cl_amd_printf : enable

//The number of consecutive floats served by one channel
#define OFFSET      64

//LEN = OFFSET * nbWorkItems (=16)
#define LEN		  1024

//JUMP = LEN - OFFSET
#define JUMP       960

/*
Basic idea:
To have two workitems per channel, to keep the channels busy.
The workitems MUST run on different CU's, to avoid stalling by other
workitems in the same wavefront.
*/
__attribute__((reqd_work_group_size(1,1,1)))
__kernel void Sum16(__global const float *array, const int n, __global float *sum)
{ 	
	int idx = get_group_id(0) * OFFSET;
	int startLastBlock = (n/OFFSET)*OFFSET;
	int startLast4Floats = (n/4) * 4;
	
	float4 s = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
	
	//Sum up a whole block
	while(idx < startLastBlock)
	{		
		#pragma unroll
		for(int i = 0; i < OFFSET; i+=4){
			s.x += array[idx + 0];
			s.y += array[idx + 1];
			s.z += array[idx + 2];
			s.w += array[idx + 3];
			idx += 4;
		}
		
		idx += JUMP;
	}
	
	//Sum up the remaining float4s, if they are in our block.
	//There is only one workitem for which: startLastBlock = idx < n < endLastBlock
	if(idx == startLastBlock)
	{
		while(idx < startLast4Floats)
		{
			s.x += array[idx + 0];
			s.y += array[idx + 1];
			s.z += array[idx + 2];
			s.w += array[idx + 3];
			idx += 4;
		}
		
		//Sum up the remaining floats
		while(idx < n)
		{
			s.x += array[idx];
			idx++;
		}
	}
	
	sum[get_group_id(0)] = s.x + s.y + s.z + s.w;
}
