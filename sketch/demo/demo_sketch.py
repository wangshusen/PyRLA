# Demo of Matrix Sketching
#
# Matrix A has more columns than rows.
# Matrix C is the sketch C = A * S,
# where S is a sketching matrix.
# We use SRFT, count sketch, and (shrinked) leverage score sampling for example.

import numpy
import matplotlib.pyplot as plt
import sys

PyRLA_dir = '../../'
sys.path.append(PyRLA_dir)

from sketch import *

def demo_srft(a_mat, s_vec, repeat_int):
    '''
    Demo of SRFT.
    
    Input
        a_mat: m-by-n (m<n) NumPy matrix A;
        s_vec: sketch sizes (int) in the ascending order;
        repeat_int: number of repeats.
    
    Output
        err_vec: each entry is the averaged relative spectral norm error 
                || A * A^T - (A * S) * (A * S)^T ||_2 / || A * A^T ||_2
                corresponding to a sketch size;
                its dimension is the same to s_vec.
    '''
    aa_mat = numpy.dot(a_mat, a_mat.T)
    aa_norm = numpy.linalg.norm(aa_mat, ord=2)
    
    num_s_int = len(s_vec)
    err_vec = numpy.zeros(num_s_int)
    for j in range(num_s_int):
        s_int = s_vec[j]
        err = 0
        for i in range(repeat_int):
            c_mat = srft.srft(a_mat, s_int) # SRFT can be replaced by any other sketching method
            cc_mat = numpy.dot(c_mat, c_mat.T)
            err += numpy.linalg.norm(aa_mat - cc_mat, ord=2)
            
        err_vec[j] = err / repeat_int / aa_norm
        print('sketch size = ' + str(s_int) + ', relative spectral norm error = ' + str(err_vec[j]))
    
    return err_vec

def demo_countsketch(a_mat, s_vec, repeat_int):
    '''
    Demo of Count Sketch.
    
    Input
        a_mat: m-by-n (m<n) NumPy matrix A;
        s_vec: sketch sizes (int) in the ascending order;
        repeat_int: number of repeats.
    
    Output
        err_vec: each entry is the averaged relative spectral norm error 
                || A * A^T - (A * S) * (A * S)^T ||_2 / || A * A^T ||_2
                corresponding to a sketch size;
                its dimension is the same to s_vec.
    '''
    aa_mat = numpy.dot(a_mat, a_mat.T)
    aa_norm = numpy.linalg.norm(aa_mat, ord=2)
    
    num_s_int = len(s_vec)
    err_vec = numpy.zeros(num_s_int)
    for j in range(num_s_int):
        s_int = s_vec[j]
        err = 0
        for i in range(repeat_int):
            c_mat = countsketch.countsketch(a_mat, s_int) # countsketch can be replaced by any other sketching method
            cc_mat = numpy.dot(c_mat, c_mat.T)
            err += numpy.linalg.norm(aa_mat - cc_mat, ord=2)
            
        err_vec[j] = err / repeat_int / aa_norm
        print('sketch size = ' + str(s_int) + ', relative spectral norm error = ' + str(err_vec[j]))
    
    return err_vec
            

def demo_sampling(a_mat, s_vec, repeat_int, lev_vec):
    '''
    Demo of Random Sampling.
    
    Input
        a_mat: m-by-n (m<n) NumPy matrix A;
        s_vec: sketch sizes (int) in the ascending order;
        repeat_int: number of repeats.
    
    Output
        err_vec: each entry is the averaged relative spectral norm error 
                || A * A^T - (A * S) * (A * S)^T ||_2 / || A * A^T ||_2
                corresponding to a sketch size;
                its dimension is the same to s_vec.
    '''
    aa_mat = numpy.dot(a_mat, a_mat.T)
    aa_norm = numpy.linalg.norm(aa_mat, ord=2)
    
    num_s_int = len(s_vec)
    err_vec = numpy.zeros(num_s_int)
    for j in range(num_s_int):
        s_int = s_vec[j]
        err = 0
        for i in range(repeat_int):
            c_mat = leverage.col_sample(a_mat, s_int, lev_vec)[1] # lev sampling can be replaced by any other sketching method
            cc_mat = numpy.dot(c_mat, c_mat.T)
            err += numpy.linalg.norm(aa_mat - cc_mat, ord=2)
            
        err_vec[j] = err / repeat_int / aa_norm
        print('sketch size = ' + str(s_int) + ', relative spectral norm error = ' + str(err_vec[j]))
    
    return err_vec
            
        
if __name__ == '__main__':
    # load real-world data
    rawdata_mat = numpy.load(PyRLA_dir + 'data/YearPredictionMSD.npy', mmap_mode='r')
    rawdata_mat = rawdata_mat[0:10000, :]
    x_mat = rawdata_mat[:, 1:].T
    m_int, n_int = x_mat.shape
    
    # parameters
    s_vec = m_int * numpy.array([2, 4, 7, 11, 16, 20, 25, 30])
    repeat_int = 20
    
    # evaluate SRFT
    print('###################### SRFT ######################')
    err_srft_vec = demo_srft(x_mat, s_vec, repeat_int)
    
    # evaluate count sketch
    print('################## count sketch ##################')
    err_count_vec = demo_countsketch(x_mat, s_vec, repeat_int)
    
    # evaluate leverage score sampling
    print('############# leverage score sampling ############')
    lev_vec = leverage.lev_approx(x_mat, 10)
    lev_vec /= sum(lev_vec)
    err_lev_vec = demo_sampling(x_mat, s_vec, repeat_int, lev_vec)
    
    # evaluate shrinked leverage score sampling
    print('######## shrinked leverage score sampling ########')
    lev_vec = leverage.lev_approx(x_mat, 10)
    lev_vec /= sum(lev_vec)
    lev_vec = (lev_vec + 1 / n_int)
    lev_vec /= sum(lev_vec)
    err_slev_vec = demo_sampling(x_mat, s_vec, repeat_int, lev_vec)

    # plot of relative spectral norm errors
    print('################## plotting... ##################')
    fig = plt.figure()
    line1, = plt.plot(s_vec, err_srft_vec, color='b', linestyle='-', marker='+', linewidth=1)
    line2, = plt.plot(s_vec, err_count_vec, color='k', linestyle=':', marker='v', linewidth=2.5)
    line3, = plt.plot(s_vec, err_lev_vec, color='r', linestyle='-.', marker='s', linewidth=2)
    line4, = plt.plot(s_vec, err_slev_vec, color='g', linestyle='--', marker='^', linewidth=1.5)
    plt.xlabel('sketch size')
    plt.ylabel('relative spectral norm error')
    #plt.tight_layout(h_pad=1.0)
    plt.legend([line1, line2, line3, line4], ['SRFT', 'count sketch', 'leverage sampling', 'shrinked leverage sampling'])
    plt.show()
    