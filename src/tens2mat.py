import numpy as np

def tens2mat(T, mode_row):
    '''
    Matlab doc, this function is simplified
    
    %TENS2MAT Matricize a tensor.
    %   M = tens2mat(T,mode_row,mode_col) matricizes a tensor T into a matrix M
    %   of dimensions prod(size_tens(mode_row))-by-prod(size_tens(mode_col)),
    %   where size_tens is equal to size(T). The columns (rows) of M are
    %   obtained by fixing the indices of T corresponding to mode_col
    %   (mode_row) and looping over the remaining indices in the order mode_row
    %   (mode_col). E.g., if A and B are two matrices and T = cat(3,A,B), then
    %   tens2mat(T,1:2,3) is the matrix [A(:) B(:)].
    %
    %   M = tens2mat(T,mode_row) matricizes a tensor T, where mode_col is
    %   chosen as the sequence [1:ndims(T)]\mode_row.
    %
    %   M = tens2mat(T,[],mode_col) matricizes a tensor T, where mode_row is
    %   chosen as the sequence [1:ndims(T)]\mode_col.
    %
    %   See also mat2tens.
    
    %   Authors: Laurent Sorber (Laurent.Sorber@cs.kuleuven.be)
    %            Marc Van Barel (Marc.VanBarel@cs.kuleuven.be)
    %            Lieven De Lathauwer (Lieven.DeLathauwer@kuleuven-kulak.be)
    '''


    size_tens_tuple = T.shape
    N = T.ndim
    
    size_row = size_tens_tuple[mode_row];
    size_col = T.size / size_row;

    mode_col = complement(mode_row,N)
    
    M = np.transpose(T, [mode_row] + mode_col)
    M = M.reshape((size_row, size_col), order='F')
    return M
  
def complement(mode_row,N):
    return range(0, mode_row) + range(mode_row + 1,N);
    