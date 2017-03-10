import numpy
import sys

PyRLA_dir = '../../'
sys.path.append(PyRLA_dir)

import sketch.countsketch as cs

def lev_exact(a_mat):
    '''
    Compute Exact Column Leverage Scores
    
    Input
        a_mat: m-by-n dense matrix A.
    
    Output
        lev_vec: n-dim vector containing the exact leverage scores
    '''
    n_int = a_mat.shape[1]
    _ , _, v_mat = numpy.linalg.svd(a_mat, full_matrices=False)
    lev_vec = numpy.sum(v_mat ** 2, axis=0)
    return lev_vec
    

    
def lev_approx(a_mat, oversampling_int=5):
    '''
    Compute Approximate Exact Column Leverage Scores
    
    Input
        a_mat: m-by-n dense matrix A;
        oversampling_int: over-sampling parameter (big over-sampling parameter leads to higher cost and better accuracy).
    
    Output
        lev_vec: n-dim vector containing the approximate leverage scores
        
    Procedure
        1. sketch size: s_int = m_int * oversampling_int
        2. draw m-by-s sketch B = A * S, where S is n-by-s count sketch matrix
        3. compute the SVD B = U * Sig * V
        4. let T = Sig^{-1} * U^T
        5. Y = T * A
        6. return the n column leverage scores of Y
    '''
    m_int, n_int = a_mat.shape
    s_int = min(m_int * oversampling_int, int(n_int / 2))
    b_mat = cs.countsketch(a_mat, s_int)
    u_mat, sig_vec, _ = numpy.linalg.svd(b_mat, full_matrices=False)
    t_mat = u_mat.T / sig_vec.reshape(len(sig_vec), 1)
    y_mat = numpy.dot(t_mat, a_mat)
    lev_vec = numpy.sum(y_mat ** 2, axis=0)
    return lev_vec


def lev_approx_bigm(a_mat, oversampling_int=5):
    '''
    Compute Approximate Exact Column Leverage Scores
    
    This algorithm is useful only if m_int is big
    
    Input
        a_mat: m-by-n dense matrix A;
        oversampling_int: over-sampling parameter (big over-sampling parameter leads to higher cost and better accuracy).
    
    Output
        lev_vec: n-dim vector containing the approximate leverage scores
        
    Procedure
        1. sketch size: s_int = m_int * oversampling_int
        2. draw m-by-s sketch B = A * S, where S is n-by-s count sketch matrix
        3. compute the SVD B = U * Sig * V
        4. let T = Sig^{-1} * U^T
        5. let p = O(log n) and generate p-by-m Gaussian projection matrix P
        5. Y = (P * T) * A
        6. return the n column leverage scores of Y
    '''
    m_int, n_int = a_mat.shape
    
    # p_int must be smaller than m_int
    p_int = int(m_int / 2) # can be tuned
    
    s_int = min(m_int * oversampling_int, int(n_int / 2))
    b_mat = cs.countsketch(a_mat, s_int)
    u_mat, sig_vec, _ = numpy.linalg.svd(b_mat, full_matrices=False)
    t_mat = u_mat.T / sig_vec.reshape(len(sig_vec), 1)
    
    p_mat = numpy.random.randn(p_int, m_int) / numpy.sqrt(p_int)
    t_mat = numpy.dot(p_mat, t_mat)
    
    y_mat = numpy.dot(t_mat, a_mat)
    lev_vec = numpy.sum(y_mat ** 2, axis=0)
    return lev_vec
    

def col_sample(a_mat, s_int, prob_vec):
    '''
    Random Sampling according to A Given Distribution
    
    Input
        a_mat: m-by-n dense matrix A;
        s_int: sketch size;
        prob_vec: n-dim vector, containing the sampling probabilities.
        
    Output
        idx_vec: n-dim vector containing the indices sampled from {1, 2, ..., n};
        c_mat: m-by-s sketch containing scaled columns of A.
    '''
    n_int = a_mat.shape[1]
    prob_vec /= sum(prob_vec)
    idx_vec = numpy.random.choice(n_int, s_int, replace=False, p=prob_vec)
    scaling_vec = numpy.sqrt(s_int * prob_vec[idx_vec]) + 1e-10
    c_mat = a_mat[:, idx_vec] / scaling_vec.reshape(1, len(scaling_vec))
    return idx_vec, c_mat
