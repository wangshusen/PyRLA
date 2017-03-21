import numpy
import sys

PyRLA_dir = '../../'
sys.path.append(PyRLA_dir)

from sketch import *

def sketched_lsr(x_mat, y_mat, sketch_size=None, sketch_type='count'):
    '''
    Sketched Least Squares Regression
    Alternative of numpy.linalg.lstsq(x_mat, y_mat)
    
    Input
        x_mat: n-by-d feature matrix;
        y_mat: n-by-m response matrix;
        sketch_size: s/d (real number greater than 1), where s is the sketch size;
        sketch_type: can be 'srft' or 'count'.
        
    Output
        w_mat: d-by-m solution;
        obj_val: objective function value (1/n) * ||X W - Y||_F^2.
    '''
    
    n_int, d_int = x_mat.shape
    
    if sketch_size is None:
        s_int = int(5 * d_int)
    else:
        s_int = int(sketch_size * d_int)
        
    if sketch_type == 'srft':
        sx_mat, sy_mat = srft.srft2(x_mat.T, y_mat.T, s_int)
        w_mat = numpy.linalg.lstsq(sx_mat.T, sy_mat.T)[0]
    elif sketch_type == 'count':
        sx_mat, sy_mat = countsketch.countsketch2(x_mat.T, y_mat.T, s_int)
        w_mat = numpy.linalg.lstsq(sx_mat.T, sy_mat.T)[0]
    elif sketch_type == 'leverage':
        lev_approx_vec = leverage.lev_approx(x_mat.T)
        prob_vec = lev_approx_vec / sum(lev_approx_vec)
        idx_vec = numpy.random.choice(n_int, s_int, replace=False, p=prob_vec)
        scaling_vec = numpy.sqrt(s_int * prob_vec[idx_vec]) + 1e-10
        sx_mat = x_mat[idx_vec, :] / scaling_vec.reshape(len(scaling_vec), 1)
        sy_mat = y_mat[idx_vec, :] / scaling_vec.reshape(len(scaling_vec), 1)
        w_mat = numpy.linalg.lstsq(sx_mat, sy_mat)[0]
    elif sketch_type == 'shrink':
        lev_approx_vec = leverage.lev_approx(x_mat.T)
        prob_vec = lev_approx_vec / sum(lev_approx_vec) + 1 / n_int
        prob_vec /= sum(prob_vec)
        idx_vec = numpy.random.choice(n_int, s_int, replace=False, p=prob_vec)
        scaling_vec = numpy.sqrt(s_int * prob_vec[idx_vec]) + 1e-10
        sx_mat = x_mat[idx_vec, :] / scaling_vec.reshape(len(scaling_vec), 1)
        sy_mat = y_mat[idx_vec, :] / scaling_vec.reshape(len(scaling_vec), 1)
        w_mat = numpy.linalg.lstsq(sx_mat, sy_mat)[0]
    
    residual = numpy.dot(x_mat, w_mat) - y_mat
    obj_val = numpy.sum(residual ** 2) / n_int
    return w_mat, obj_val
    
    