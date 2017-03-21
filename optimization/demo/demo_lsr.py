# Demo of the Sketched Least Squares Regression

import numpy
import matplotlib.pyplot as plt
import sys

PyRLA_dir = '../../'
sys.path.append(PyRLA_dir)

import optimization.lsr as lsr



def demo_lsr(x_mat, y_mat, sketch_type, sketch_size_vec, repeat=10):
    d_int, n_int = x_mat.shape
    
    # optimal solution
    w_opt_mat = numpy.dot(numpy.linalg.pinv(x_mat), y_mat)
    residual = numpy.dot(x_mat, w_opt_mat) - y_mat
    opt_obj_val = numpy.sum(residual ** 2) / n_int
    
    s_vec_len = len(sketch_size_vec)
    obj_vec = numpy.zeros(s_vec_len)
    dist_vec1 = numpy.zeros(s_vec_len)
    dist_vec2 = numpy.zeros(s_vec_len)
    
    for i in range(s_vec_len):
        obj = 0
        dist1 = 0
        dist2 = 0
        for j in range(repeat):
            w_sketch_mat, obj_val = lsr.sketched_lsr(x_mat, y_mat, sketch_size=sketch_size_vec[i], sketch_type=sketch_type)
            obj += obj_val
            dist_mat = w_sketch_mat - w_opt_mat
            dist1 += numpy.sum(dist_mat ** 2)
            dist2 += numpy.sum(numpy.dot(x_mat, dist_mat) ** 2)
        obj_vec[i] = obj / repeat
        dist_vec1[i] = dist1 / repeat
        dist_vec2[i] = dist2 / repeat
    
    return obj_vec, dist_vec1, dist_vec2 / n_int

def plot_dist1(sketch_size_vec, srft_vec, count_vec, lev_vec, shrink_vec):
    fig = plt.figure(figsize=(9, 7))
    line1, = plt.plot(sketch_size_vec, srft_vec, color='r', linestyle='-', linewidth=3)
    line2, = plt.plot(sketch_size_vec, count_vec, color='b', linestyle='--', linewidth=3)
    line3, = plt.plot(sketch_size_vec, lev_vec, color='g', linestyle='-.', linewidth=3)
    line4, = plt.plot(sketch_size_vec, shrink_vec, color='c', linestyle='-.', linewidth=4)
    
    plt.legend([line1, line2, line3, line4], ['SRFT', 'Count Sketch', 'Leverage Sampling', 'Shrinked Leverage Sampling'], fontsize=20)
    plt.xlabel('s / d', fontsize=26)
    plt.ylabel(r"$\| \tilde{\mathbf{w}} - \mathbf{w}^\star \|_2^2 \: / \: \| \mathbf{w}^\star \|_2^2$", fontsize=24)
    plt.xticks(fontsize=22) 
    plt.yticks(fontsize=22) 
    #plt.axis([1, numpy.ceil(max(sketch_size_vec)), 1, numpy.ceil(max(lev_count_vec))])
    plt.tight_layout()
    
    fig.savefig(PyRLA_dir + 'output/sketch_lsr_dist1.pdf', format='pdf', dpi=1200)
    plt.show()
    
    
def plot_dist2(sketch_size_vec, srft_vec, count_vec, lev_vec, shrink_vec):
    fig = plt.figure(figsize=(9, 7))
    line1, = plt.plot(sketch_size_vec, srft_vec, color='r', linestyle='-', linewidth=3)
    line2, = plt.plot(sketch_size_vec, count_vec, color='b', linestyle='--', linewidth=3)
    line3, = plt.plot(sketch_size_vec, lev_vec, color='g', linestyle='-.', linewidth=3)
    line4, = plt.plot(sketch_size_vec, shrink_vec, color='c', linestyle='-.', linewidth=4)
    
    plt.legend([line1, line2, line3, line4], ['SRFT', 'Count Sketch', 'Leverage Sampling', 'Shrinked Leverage Sampling'], fontsize=20)
    plt.xlabel('s / d', fontsize=26)
    plt.ylabel(r"$ \| \mathbf{X} \tilde{\mathbf{w}} -  \mathbf{X}  \mathbf{w}^\star \|_2^2 \: / \: \|  \mathbf{X}  \mathbf{w}^\star \|_2^2$", fontsize=24)
    plt.xticks(fontsize=22) 
    plt.yticks(fontsize=22) 
    #plt.axis([1, numpy.ceil(max(sketch_size_vec)), 1, numpy.ceil(max(lev_count_vec))])
    plt.tight_layout()
    
    fig.savefig(PyRLA_dir + 'output/sketch_lsr_dist2.pdf', format='pdf', dpi=1200)
    plt.show()
    
    
def plot_obj(sketch_size_vec, srft_vec, count_vec, lev_vec, shrink_vec):
    fig = plt.figure(figsize=(9, 7))
    line1, = plt.plot(sketch_size_vec, srft_vec, color='r', linestyle='-', linewidth=3)
    line2, = plt.plot(sketch_size_vec, count_vec, color='b', linestyle='--', linewidth=3)
    line3, = plt.plot(sketch_size_vec, lev_vec, color='g', linestyle='-.', linewidth=3)
    line4, = plt.plot(sketch_size_vec, shrink_vec, color='c', linestyle='-.', linewidth=4)
    #line5, = plt.plot(sketch_size_vec, opt_obj_val * numpy.ones(len(sketch_size_vec)), color='k', linestyle=':', linewidth=3)
    
    plt.legend([line1, line2, line3, line4], ['SRFT', 'Count Sketch', 'Leverage Sampling', 'Shrinked Leverage Sampling'], fontsize=20)
    plt.xlabel('s / d', fontsize=26)
    plt.ylabel(r"$f(\tilde{\mathbf{w}}) \: / \: f({\mathbf{w}}^\star)$", fontsize=24)
    plt.xticks(fontsize=22) 
    plt.yticks(fontsize=22)
    
    plt.axis([2, 16, 1, 1.5])
    plt.tight_layout()
    
    fig.savefig(PyRLA_dir + 'output/sketch_lsr_obj.pdf', format='pdf', dpi=1200)
    plt.show()
    
    

if __name__ == '__main__':
    n_int = 100000 # can be tuned; do not exceed #rows of rawdata_mat
    repeat = 100 # can be tuned
    
    rawdata_mat = numpy.load(PyRLA_dir + 'data/YearPredictionMSD.npy', mmap_mode='r')
    rawdata_mat = rawdata_mat[0:50000, :]
    x_mat = rawdata_mat[:, 1:]
    n_int, d_int = x_mat.shape
    y_vec = rawdata_mat[:, 0].reshape((n_int, 1))
    
    # optimal solution
    w_opt_vec = numpy.dot(numpy.linalg.pinv(x_mat), y_vec)
    residual = numpy.dot(x_mat, w_opt_vec) - y_vec
    opt_obj_val = numpy.sum(residual ** 2) / n_int
    
    dist_opt1 = numpy.sum(w_opt_vec ** 2)
    dist_opt2 = numpy.sum(numpy.dot(x_mat, w_opt_vec) ** 2) / n_int
    
    #sketch_size_vec = [6, 7, 8, 9, 10]
    sketch_size_vec = [3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    obj_srft_vec, dist_srft_vec1, dist_srft_vec2 = demo_lsr(x_mat, y_vec, 'srft', sketch_size_vec, repeat=repeat)
    obj_count_vec, dist_count_vec1, dist_count_vec2 = demo_lsr(x_mat, y_vec, 'count', sketch_size_vec, repeat=repeat)
    obj_lev_vec, dist_lev_vec1, dist_lev_vec2 = demo_lsr(x_mat, y_vec, 'leverage', sketch_size_vec, repeat=repeat)
    obj_shrink_vec, dist_shrink_vec1, dist_shrink_vec2 = demo_lsr(x_mat, y_vec, 'shrink', sketch_size_vec, repeat=repeat)
    
    # plot squared l2 distance
    plot_dist1(sketch_size_vec, dist_srft_vec1 / dist_opt1, dist_count_vec1 / dist_opt1, dist_lev_vec1 / dist_opt1, dist_shrink_vec1 / dist_opt1)
    plot_dist2(sketch_size_vec, dist_srft_vec2 / dist_opt2, dist_count_vec2 / dist_opt2, dist_lev_vec2 / dist_opt2, dist_shrink_vec2 / dist_opt2)

    # plot objective function value
    plot_obj(sketch_size_vec, obj_srft_vec / opt_obj_val, obj_count_vec / opt_obj_val, obj_lev_vec / opt_obj_val, obj_shrink_vec / opt_obj_val)


