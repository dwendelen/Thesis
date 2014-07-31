__attribute__((reqd_work_group_size(8, 8, 1)))
__kernel void KernelG1(__global const double2 *T,
    __global double2 *G1, __global const double2 *U2, __global const double2 *U3,
    const int I2, const int I3)
{
    double2 temp;
    double2 u2;
    
    double2[32] U3Cache;
    int nbOfBlocksCached;
    int idxCache;
    
    double2 sum = 0;
    
    //  idxTFirstCacheItem = idxT1
    int idxTFirstCacheItem = get_global_id(1);
    int idxT;
    
    //  idxU2 =       idxR       * I2;
    int idxU2 = get_global_id(0) * I2;
    int idxU3 = get_global_id(0) * I3;
    
    int limitU2 = idxU2 + I2;
    int limitU3 = idxU3 + I3;
    
    int jumpIdxTMode3 = 2*I1*I2;
    int jumpIdxTMode2 = I1;
    
    //  jumpIdxTForNextCache = 32*jumpIdxTMode3 - 2*jumpIdxTMode2*I2
    //  jumpIdxTForNextCache = 32*jumpIdxTMode3 - 2*I1*I2
    //  jumpIdxTForNextCache = 32*jumpIdxTMode3 - jumpIdxTMode3
    int jumpIdxTForNextCache = 31*jumpIdxTMode3;
    
    while(idxU3 < limitU3)
    {
    	sizeCache = 0;
    	
    	//Cache 64 doubles in blocks of 16, if available
    	#pragma unroll
    	for(int i = 0; i < 4; i++)
    	{	
    		if(idxU3 < limitU3)
    			break;
    			
	    	#pragma unroll
	    	for(int j = 0; j < 8; j++)
	    	{
	        	U3Cache[j] = U3[idxU3++];
	    	}
        	sizeCache += 8;
        }
        
        for(idxU2 = 0; idxU2 < limitU2; idxU2++)
        {
            u2 = U2[idxU2];
            
            idxT = idxTFirstCacheItem;
            for(idxCache = 0; idxCache < sizeCache; idxCache += 8)
            {
            	#pragma unroll
            	for(int j = 0; j < 8; j++)
		    	{
		        	temp = u2 * U3Cache[j].x;
		        	sum += temp.x * T[tIdx];
		        	
		        	temp = u2 * U3Cache[j].y;
		        	sum += temp.x * T[tIdx + jumpIdxTMode2];
            		idxT += jumpIdxTMode3;
		    	}
            }
            idxTFirstCacheItem += 2*jumpIdxTMode2;
        }
        idxTFirstCacheItem += jumpIdxTForNextCache;
    }
    
    //  idxU1 =       idxT1      + I0 * idxR
    int idxU1 = get_global_id(1) + I0 * get_global_id(0);
    G1[idxU1] = sum;
}
