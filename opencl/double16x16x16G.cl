__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void KernelG1(__global const double2 *T, __global double2 *G1, __global const double2 *U2, __global const double2 *U3, const int I2, const int I3)
{
    double2 temp;
    double2 u2;
    
    double2 U3Cache[32];
    int sizeCache;
    int idxCache;
    
    double2 sum = 0;
    
    int idxTFirstCacheItem = get_global_id(1);
    int idxT;
    int idxR = get_global_id(0);
    int I1 = get_global_size(1);
    
    int startIdxU2 = idxR * I2;
    int idxU2;
    int idxU3 = idxR * I3;
    
    int limitU2 = idxU2 + I2;
    int limitU3 = idxU3 + I3;
    
    int jumpIdxTMode2 = I1;
    int jumpIdxTMode3 = 2 * I1 * I2;
    
    //  jumpIdxTForNextCache = 64*jumpIdxTMode3 - 2*jumpIdxTMode2*I2
    //  jumpIdxTForNextCache = 64*jumpIdxTMode3 - 2*I1*I2
    //  jumpIdxTForNextCache = 64*jumpIdxTMode3 - jumpIdxTMode3
    int jumpIdxTForNextCache = 63*jumpIdxTMode3;
    
    while(idxU3 < limitU3)
    {
    	sizeCache = 0;
    	
    	//Cache 64 doubles in blocks of 16, if available
    	#pragma unroll
    	for(int i = 0; i < 4; i++)
    	{	
    		if(idxU3 >= limitU3)
    			break;
    			
	    	#pragma unroll
	    	for(int j = 0; j < 8; j++)
	    	{
	        	U3Cache[j] = U3[idxU3++];
	    	}
        	sizeCache += 8;
        }
        
        for(idxU2 = startIdxU2; idxU2 < limitU2; idxU2++)
        {
            u2 = U2[idxU2];
            
            idxT = idxTFirstCacheItem;
            for(int idxCache = 0; idxCache < sizeCache; )
            {
            	#pragma unroll
            	for(int j = 0; j < 8; j++)
		    	{
		        	temp = u2 * U3Cache[idxCache].x;
		        	sum += temp.x * T[idxT];
		        	sum += temp.y * T[idxT + jumpIdxTMode2];
		        	idxT += jumpIdxTMode3;
		        	
		        	temp = u2 * U3Cache[idxCache++].y;
		        	sum += temp.x * T[idxT];
					sum += temp.y * T[idxT + jumpIdxTMode2];
            		idxT += jumpIdxTMode3;
		    	}
            }
            idxTFirstCacheItem += 2*jumpIdxTMode2;
        }
        idxTFirstCacheItem += jumpIdxTForNextCache;
    }
    
    //  idxG1 =       idxT1      +         I0         *     idxR
    int idxG1 = get_global_id(1) + get_global_size(1) * get_global_id(0);
    G1[idxG1] = sum;
}

__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void KernelG2(__global const double2 *T,
    __global double2 *U1, __global const double2 *G2, __global const double2 *U3,
    const int I1, const int I3)
    {
    }
    
    __attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void KernelG3(__global const double2 *T,
    __global double2 *U1, __global const double2 *U2, __global const double2 *G3,
    const int I1, const int I2)
    {
    }
