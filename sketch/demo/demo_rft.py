# Demo of Randomized Fourier Transform (RFT)
#
# Matrix A has more columns than rows.
# Apply RFT to the rows of A: A*D*F,
# where D has random signs as diagonal entries, and F is real-FFT mattrix.
# The column coherence of A*D*F is much smaller than that of A.

import numpy
import matplotlib.pyplot as plt
import sys

PyRLA_dir = '../../'
sys.path.append(PyRLA_dir)

from sketch import srft

def col_lev(a_mat):
    '''
    Compute the Column Leverage Scores
    
    Input
        a_mat: m-by-n (m<n) NumPy matrix A
    Output
        lev_vec: n-dim vector containing the leverage scores of A
    '''
    _ , _, v_mat = numpy.linalg.svd(a_mat, full_matrices=False)
    lev_vec = numpy.sum(v_mat ** 2, axis=0)
    return lev_vec
        
if __name__ == '__main__':
    # load real-world data
    rawdata_mat = numpy.load(PyRLA_dir + 'data/YearPredictionMSD.npy', mmap_mode='r')
    rawdata_mat = rawdata_mat[0:2000, :]
    x_mat = rawdata_mat[:, 1:].T
    m_int, n_int = x_mat.shape

    # X is the input matrix with much more columns than rows
    # compute the column leverage scores of X
    lev_x_vec = col_lev(x_mat)
    
    # compute the randomized Fourier transform C = X * D * F
    # where D is a diagonal matrix whose diagonal entries are random signs,
    # and F is the real FFT matrix.
    sign_vec = numpy.random.choice(2, n_int) * 2 - 1
    c_mat = srft.realfft_row(x_mat * sign_vec.reshape(1, n_int))
    
    # compute the column leverage scores of C
    lev_c_vec = col_lev(c_mat)
    
    # compute the column coherence
    x_coherence = n_int / m_int * max(lev_x_vec)
    c_coherence = n_int / m_int * max(lev_c_vec)
    
    print('The column coherence is ' + str(x_coherence))
    print('After the randomized Fourier transform,')
    print('the column coherence is ' + str(c_coherence))
    
    # bar plots of the leverage scores
    idx = numpy.arange(n_int)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax1.bar(idx, lev_x_vec, color='b', edgecolor='b')
    ax1.axis([0, n_int, 0, 1])
    plt.xlabel('indices')
    plt.ylabel('leverage score')
    plt.title('leverage scores (before randomized Fourier transform)', color='b')
    ax2 = fig.add_subplot(212)
    ax2.bar(idx, lev_c_vec, color='r', edgecolor='r')
    ax2.axis([0, n_int, 0, 1])
    plt.xlabel('indices')
    plt.ylabel('leverage score')
    plt.title('leverage scores (after randomized Fourier transform)', color='r')
    plt.tight_layout(h_pad=1.0)
    plt.show()
    