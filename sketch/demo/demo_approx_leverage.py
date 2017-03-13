# Demo of the Effect of Precondition
#
# We compare the convergence of CG with/without precondition

import numpy
import matplotlib.pyplot as plt
import sys

PyRLA_dir = '../../'
sys.path.append(PyRLA_dir)

import sketch.leverage as lev

def demo_approxlev(a_mat, sketch_type, sketch_size_vec, repeat=10):
    d_int, n_int = a_mat.shape
    lev_vec = lev.lev_exact(a_mat)
    
    s_vec_len = len(sketch_size_vec)
    err_vec = numpy.zeros(s_vec_len)
    
    for i in range(s_vec_len):
        err = 0
        for j in range(repeat):
            approx_lev_vec = lev.lev_approx(a_mat, sketch_size=sketch_size_vec[i], sketch_type=sketch_type)
            ratio_vec = approx_lev_vec / lev_vec
            err += max(ratio_vec) / min(ratio_vec)
        err_vec[i] = err / repeat
    
    return err_vec

def plot_approx_error(sketch_size_vec, lev_srft_vec, lev_count_vec):
    fig = plt.figure(figsize=(9, 5))
    line1, = plt.plot(sketch_size_vec, lev_srft_vec, color='r', linestyle='-', linewidth=3)
    line2, = plt.plot(sketch_size_vec, lev_count_vec, color='b', linestyle='--', linewidth=3)
    
    plt.legend([line1, line2], ['SRFT', 'Count Sketch'], fontsize=21)
    plt.xlabel('s / d', fontsize=26)
    plt.ylabel('Approximation Error', fontsize=24)
    plt.xticks(fontsize=22) 
    plt.yticks(fontsize=22) 
    plt.axis([1, numpy.ceil(max(sketch_size_vec)), 1, numpy.ceil(max(lev_count_vec))])
    plt.tight_layout()
    
    fig.savefig(PyRLA_dir + 'output/lev_approx.pdf', format='pdf', dpi=1200)
    plt.show()
    
    
def demo_approxlev_fast(a_mat, sketch_type, sketch_size, speedup_vec, repeat=10):
    d_int, n_int = a_mat.shape
    lev_vec = lev.lev_exact(a_mat)
    
    speedup_vec_len = len(speedup_vec)
    err_vec = numpy.zeros(speedup_vec_len)
    
    for i in range(speedup_vec_len):
        err = 0
        for j in range(repeat):
            approx_lev_vec = lev.lev_approx_fast(a_mat, sketch_size=sketch_size, sketch_type=sketch_type, speedup=speedup_vec[i])
            ratio_vec = approx_lev_vec / lev_vec
            err += max(ratio_vec) / min(ratio_vec)
        err_vec[i] = err / repeat
    
    return err_vec
    
    
def plot_approx_error2(speedup_vec, err_s2_vec, err_s5_vec, err_s10_vec):
    fig = plt.figure(figsize=(9, 5))
    line1, = plt.plot(1 / speedup_vec, err_s2_vec, color='r', linestyle='-', linewidth=3)
    line2, = plt.plot(1 / speedup_vec, err_s5_vec, color='b', linestyle='--', linewidth=3)
    line3, = plt.plot(1 / speedup_vec, err_s10_vec, color='g', linestyle='-.', linewidth=3)
    
    plt.legend([line1, line2, line3], ['s = 2d', 's = 5d', 's = 10d'], fontsize=21)
    plt.xlabel('p / d', fontsize=26)
    plt.ylabel('Approximation Error', fontsize=24)
    plt.xticks(fontsize=22) 
    plt.yticks([1, 5, 10, 15, 20, 25, 30], fontsize=22) 
    plt.axis([numpy.min(1/speedup_vec), numpy.max(1/speedup_vec), 1, 25])
    plt.tight_layout()
    
    fig.savefig(PyRLA_dir + 'output/lev_approx_fast.pdf', format='pdf', dpi=1200)
    plt.show()

if __name__ == '__main__':
    n_int = 10000 # can be tuned; do not exceed #rows of rawdata_mat
    repeat = 20
    
    rawdata_mat = numpy.load(PyRLA_dir + 'data/YearPredictionMSD.npy', mmap_mode='r')
    rawdata_mat = rawdata_mat[0:n_int, :]
    x_mat = rawdata_mat[:, 1:].T
    
    sketch_size_vec = [1.5, 1.6, 1.8, 2, 2.5, 3, 4, 5, 6, 8, 10]
    err_count_vec = demo_approxlev(x_mat, 'count', sketch_size_vec=sketch_size_vec, repeat=repeat)
    err_srft_vec = demo_approxlev(x_mat, 'srft', sketch_size_vec=sketch_size_vec, repeat=repeat)
    #plot_approx_error(sketch_size_vec, err_srft_vec, err_count_vec)
    
    speedup_vec = numpy.array([5, 4.8, 4.6, 4.3, 4, 3.5, 3, 2.5, 2, 1.8, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1])
    err_s2_vec = demo_approxlev_fast(x_mat, 'count', 2, speedup_vec, repeat=10)
    err_s5_vec = demo_approxlev_fast(x_mat, 'count', 5, speedup_vec, repeat=10)
    err_s10_vec = demo_approxlev_fast(x_mat, 'count', 10, speedup_vec, repeat=10)
    plot_approx_error2(speedup_vec, err_s2_vec, err_s5_vec, err_s10_vec)



