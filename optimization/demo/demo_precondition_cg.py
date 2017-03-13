# Demo of the Effect of Precondition
#
# We compare the convergence of CG with/without precondition

import numpy
import matplotlib.pyplot as plt
import sys

PyRLA_dir = '../../'
sys.path.append(PyRLA_dir)

from optimization import *


def demo_cg_converge(x_mat, y_vec, sketch_type):
    # ================= optimal solution ================= #
    w_opt_vec = numpy.dot(numpy.linalg.pinv(x_mat), y_vec)

    # ============= CG without precondition ============== #
    # CG without precondition
    err_vec = cg.demo_cg(x_mat, y_vec, w_opt_vec)[2]

    # =============== CG with precondition =============== #
    # preconditioning matrix is computed by using s=1.2d
    t_mat = cg_cond.precondition(x_mat, sketch_type=sketch_type, sketch_size=1.2)
    err_cond1_vec = cg_cond.demo_cg_cond(x_mat, y_vec, t_mat, w_opt_vec)[2]
    
    # preconditioning matrix is computed by using s=2d
    t_mat = cg_cond.precondition(x_mat, sketch_type=sketch_type, sketch_size=2)
    err_cond2_vec = cg_cond.demo_cg_cond(x_mat, y_vec, t_mat, w_opt_vec)[2]

    # preconditioning matrix is computed by using s=4d
    t_mat = cg_cond.precondition(x_mat, sketch_type=sketch_type, sketch_size=4)
    err_cond4_vec = cg_cond.demo_cg_cond(x_mat, y_vec, t_mat, w_opt_vec)[2]
    
    
    # ====================== Plot ====================== #
    fig = plt.figure(figsize=(9,6))
    line1, = plt.semilogy(list(range(len(err_vec))), err_vec, color='k', linestyle='-', linewidth=2)
    line2, = plt.semilogy(list(range(len(err_cond1_vec))), err_cond1_vec, color='b', linestyle='-.', linewidth=4)
    line3, = plt.semilogy(list(range(len(err_cond2_vec))), err_cond2_vec, color='r', linestyle='--', linewidth=3)
    
    line4, = plt.semilogy(list(range(len(err_cond4_vec))), err_cond4_vec, color='g', linestyle='-', linewidth=3)
    
    if sketch == 'srft':
        title = 'SRFT'
    elif sketch == 'count':
        title = 'Count Sketch'
    plt.title(title, fontsize=30)
    
    plt.legend([line1, line2, line3, line4], ['without precondition', 'precondition (s=1.2d)', 'precondition (s=2d)', 'precondition (s=4d)'], fontsize=21)
    plt.xlabel('Number of Iterations', fontsize=26)
    plt.ylabel('Squared L2 Distance', fontsize=24)
    plt.xticks(fontsize=22) 
    plt.yticks([1e10, 1e6, 1e2, 1e-2, 1e-6, 1e-10, 1e-14, 1e-18], fontsize=22) 
    plt.tight_layout()
    
    fig.savefig(PyRLA_dir + 'output/cg_convergence_' + sketch + '.pdf', format='pdf', dpi=1200)
    plt.show()


if __name__ == '__main__':
    n_int = 100000 # do not exceed #rows of rawdata_mat
    rawdata_mat = numpy.load(PyRLA_dir + 'data/YearPredictionMSD.npy', mmap_mode='r')
    rawdata_mat = rawdata_mat[0:n_int, :]
    x_mat = rawdata_mat[:, 1:]
    y_vec = rawdata_mat[:, 0]
    
    sketch = 'count' # can be 'srft' or 'count'
    
    demo_cg_converge(x_mat, y_vec, sketch)
    
    




