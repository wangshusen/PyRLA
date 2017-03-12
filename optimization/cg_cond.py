import numpy

import numpy
import matplotlib.pyplot as plt
import sys

PyRLA_dir = '../../'
sys.path.append(PyRLA_dir)

import sketch.srft as srft
import sketch.countsketch as cs


def precondition(x_mat, sketch_type='srft', sketch_size=3):
    '''
    Compute A Preconditioning Matrix
    
    Input
        x_mat: n-by-d NumPy matrix X;
        sketch_type: 'srft' or 'count';
        sketch_size: real number larger than 1;
                    it should be set as a small number, e.g. 3;
                    big sketch_size leads to good condition number, but costs more time to compute.
    
    Output
        t_mat: d-by-d Numpy matrix T such that X * T is well-conditioned.
    '''
    n_int, d_int = x_mat.shape
    s_int = int(sketch_size * d_int)
    
    if sketch_type == 'srft':
        b_mat = srft.srft(x_mat.T, s_int)
    elif sketch_type == 'count':
        b_mat = cs.countsketch(x_mat.T, s_int)
        
    u_mat, sig_vec, _ = numpy.linalg.svd(b_mat, full_matrices=False)
    t_mat = u_mat / sig_vec.reshape(1, len(sig_vec))
    return t_mat
    
    
def cg_cond(x_mat, y_vec, cond_mat, tol=1e-16, max_iter_int=10000):
    '''
    Preconditioned Conjugate Gradient (CG) Algorithm for Least Squares Regression: argmin_w || X * w - y ||_2^2
    
    Input
        x_mat: n-by-d NumPy matrix X;
        y_vec: n-dim vector y;
        cond_mat: d-by-d preconditioned matrix T such that X * T is well-conditioned;
        tol: convergence tolerance (optional);
        max_iter_int: (>0) maximum number of iterations.
        
    Output
        w_vec: solution to the LSR problem;
        is_converged_bool: whether CG attains convergence tolerance.
    '''
    
    n_int, d_int = x_mat.shape
    y_vec = numpy.dot(x_mat.T, y_vec.reshape(n_int, 1))
    y_vec = numpy.dot(cond_mat.T, y_vec)
    
    z_vec = numpy.zeros((d_int, 1))
    w_vec = numpy.dot(cond_mat, z_vec)
        
    xw_vec = numpy.dot(x_mat, w_vec)
    r_vec = y_vec - numpy.dot(cond_mat.T, numpy.dot(x_mat.T, xw_vec))
    p_vec = r_vec
    rsold_real = numpy.sum(r_vec ** 2)
    
    is_converged_bool = False
    i = -1
    
    for i in range(max_iter_int):
        xp_vec = numpy.dot(x_mat, numpy.dot(cond_mat, p_vec))
        xp_vec = numpy.dot(cond_mat.T, numpy.dot(x_mat.T, xp_vec))
        alp_real = rsold_real / numpy.sum(p_vec * xp_vec)
        z_vec += alp_real * p_vec
        r_vec -= alp_real * xp_vec
        rsnew_real = numpy.sum(r_vec ** 2)
        
        if i % 100 == 0:
            print('Iteration ' + str(i) + ': residual=' + str(rsnew_real))
        
        if rsnew_real < tol:
            is_converged_bool = True
            print('Iteration ' + str(i) + ': residual=' + str(rsnew_real))
            break
            
        p_vec = r_vec + (rsnew_real / rsold_real) * p_vec
        rsold_real = rsnew_real
        
    w_vec = numpy.dot(cond_mat, z_vec)
    
    if not is_converged_bool:
        print('Warn: CG did not converge after ' + str(i+1) + ' iterations!')
    
    return w_vec, is_converged_bool
    
    
    
def demo_cg_cond(x_mat, y_vec, cond_mat, w_opt_vec, tol=1e-16, max_iter_int=10000):
    '''
    For Demo Usage Only!
    
    Preconditioned Conjugate Gradient (CG) Algorithm for Least Squares Regression: argmin_w || X * w - y ||_2^2
    
    Input
        x_mat: n-by-d NumPy matrix X;
        y_vec: n-dim vector y;
        w_opt_vec: optimal solution;
        cond_mat: d-by-d preconditioned matrix T such that X * T is well-conditioned;
        tol: convergence tolerance (optional);
        max_iter_int: (>0) maximum number of iterations.
        
    Output
        w_vec: solution to the LSR problem;
        is_converged_bool: whether CG attains convergence tolerance;
        tmp_err_vec: the error || X * w_opt - X * w ||_2^2 of each step.
    '''
    tmp_xw_opt_vec = numpy.dot(x_mat, w_opt_vec.reshape(len(w_opt_vec), 1)) # for test only 

    
    n_int, d_int = x_mat.shape
    y_vec = numpy.dot(x_mat.T, y_vec.reshape(n_int, 1))
    y_vec = numpy.dot(cond_mat.T, y_vec)
    
    z_vec = numpy.zeros((d_int, 1))
    w_vec = numpy.dot(cond_mat, z_vec)
        
    xw_vec = numpy.dot(x_mat, w_vec)
    r_vec = y_vec - numpy.dot(cond_mat.T, numpy.dot(x_mat.T, xw_vec))
    p_vec = r_vec
    rsold_real = numpy.sum(r_vec ** 2)
    
    is_converged_bool = False
    i = -1
    
    tmp_err_vec = numpy.zeros(max_iter_int) # for test only
    
    for i in range(max_iter_int):
        xp_vec = numpy.dot(x_mat, numpy.dot(cond_mat, p_vec))
        xp_vec = numpy.dot(cond_mat.T, numpy.dot(x_mat.T, xp_vec))
        alp_real = rsold_real / numpy.sum(p_vec * xp_vec)
        z_vec += alp_real * p_vec
        r_vec -= alp_real * xp_vec
        rsnew_real = numpy.sum(r_vec ** 2)
        
        if i % 100 == 0:
            print('Iteration ' + str(i) + ': residual=' + str(rsnew_real))
        
        tmp_dist_vec = numpy.dot(x_mat, numpy.dot(cond_mat, z_vec)) - tmp_xw_opt_vec # for test only 
        tmp_err_vec[i] = numpy.sum(tmp_dist_vec ** 2) # for test only 
        
        if rsnew_real < tol:
            is_converged_bool = True
            print('Iteration ' + str(i) + ': residual=' + str(rsnew_real))
            break
            
        p_vec = r_vec + (rsnew_real / rsold_real) * p_vec
        rsold_real = rsnew_real
        
    w_vec = numpy.dot(cond_mat, z_vec)
    
    if not is_converged_bool:
        print('Warn: CG did not converge after ' + str(i+1) + ' iterations!')
    
    tmp_err_vec = tmp_err_vec[0: i+1] # for test only
    
    return w_vec, is_converged_bool, tmp_err_vec
    
    
    
    
    