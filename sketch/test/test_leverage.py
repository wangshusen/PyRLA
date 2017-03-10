import numpy
import unittest
import sys

PyRLA_dir = '../../'
sys.path.append(PyRLA_dir)

import sketch.leverage as lev

rawdata_mat = numpy.load(PyRLA_dir + 'data/YearPredictionMSD.npy', mmap_mode='r')
rawdata_mat = rawdata_mat[0:50007, :]
x_mat = rawdata_mat[:, 1:].T
m_int, n_int = x_mat.shape
y_mat = rawdata_mat[:, 0].reshape((1, n_int))
d_int = y_mat.shape[0]


xx_mat = numpy.dot(x_mat, x_mat.T)
xx_norm = numpy.linalg.norm(xx_mat, ord='fro')
xy_mat = numpy.dot(x_mat, y_mat.T)
xy_norm = numpy.linalg.norm(xy_mat, ord='fro')


# exact leverage scores
lev_exact_vec = lev.lev_exact(x_mat)

class TestCountSketch(unittest.TestCase):
    
    def test_exact_lev_score(self):
        print('Exact leverage scores: max=' + str(max(lev_exact_vec)) + '   min=' + str(min(lev_exact_vec)))
        self.assertTrue(max(lev_exact_vec) >= 0)
        self.assertTrue(max(lev_exact_vec) <= 1)
        self.assertEqual(len(lev_exact_vec), n_int)
        
    def test_approx_lev_score(self):
        # approximate leverage scores
        lev_approx_vec = lev.lev_approx(x_mat)
        print('Approx leverage scores: max=' + str(max(lev_approx_vec)) + '   min=' + str(min(lev_approx_vec)))
        self.assertEqual(len(lev_approx_vec), n_int)
        
        repeat = 10
        
        oversampling_int = 2
        err = 0
        for i in range(repeat):
            lev_approx_vec = lev.lev_approx(x_mat, oversampling_int)
            ratio_vec = lev_approx_vec / lev_exact_vec
            err += max(ratio_vec) / min(ratio_vec)
        err1 = err / repeat
        print('Approximate ratio: ' + str(err1))
        
        oversampling_int = 4
        err = 0
        for i in range(repeat):
            lev_approx_vec = lev.lev_approx(x_mat, oversampling_int)
            ratio_vec = lev_approx_vec / lev_exact_vec
            err += max(ratio_vec) / min(ratio_vec)
        err2 = err / repeat
        print('Approximate ratio: ' + str(err2))
        
        oversampling_int = 10
        err = 0
        for i in range(repeat):
            lev_approx_vec = lev.lev_approx(x_mat, oversampling_int)
            ratio_vec = lev_approx_vec / lev_exact_vec
            err += max(ratio_vec) / min(ratio_vec)
        err3 = err / repeat
        print('Approximate ratio: ' + str(err3))
        
        self.assertTrue(err2 < err1)
        self.assertTrue(err3 < err2)
        
    def test_approx_lev_score_bigm(self):
        # approximate leverage scores
        lev_approx_vec = lev.lev_approx_bigm(x_mat)
        print('Approx leverage scores (with Gaussian projection): max=' + str(max(lev_approx_vec)) + '   min=' + str(min(lev_approx_vec)))
        self.assertEqual(len(lev_approx_vec), n_int)
        
        repeat = 10
        
        oversampling_int = 2
        err = 0
        for i in range(repeat):
            lev_approx_vec = lev.lev_approx_bigm(x_mat, oversampling_int)
            ratio_vec = lev_approx_vec / lev_exact_vec
            err += max(ratio_vec) / min(ratio_vec)
        err1 = err / repeat
        print('Approximate ratio (with Gaussian projection): ' + str(err1))
        
        oversampling_int = 4
        err = 0
        for i in range(repeat):
            lev_approx_vec = lev.lev_approx_bigm(x_mat, oversampling_int)
            ratio_vec = lev_approx_vec / lev_exact_vec
            err += max(ratio_vec) / min(ratio_vec)
        err2 = err / repeat
        print('Approximate ratio (with Gaussian projection): ' + str(err2))
        
        oversampling_int = 10
        err = 0
        for i in range(repeat):
            lev_approx_vec = lev.lev_approx_bigm(x_mat, oversampling_int)
            ratio_vec = lev_approx_vec / lev_exact_vec
            err += max(ratio_vec) / min(ratio_vec)
        err3 = err / repeat
        print('Approximate ratio (with Gaussian projection): ' + str(err3))
        
        self.assertTrue(err2 < err1)
        self.assertTrue(err3 < err2)
        
    def test_approx_lev_sampling(self):
        '''
        Test leverage score sampling.
        As the sketch size s_int increases, the approximation error should decrease.
        If the test fails, say twice in 10 tests, it is fine.
        '''
        repeat = 10
        
        s_int1 = 150
        err1 = 0
        for i in range(repeat):
            lev_vec = lev.lev_approx(x_mat)
            c_mat = lev.col_sample(x_mat, s_int1, lev_vec)[1]
            err1 += numpy.linalg.norm(xx_mat - numpy.dot(c_mat, c_mat.T), ord='fro') / xx_norm
        err1 /= repeat
        
        s_int2 = 400
        err2 = 0
        for i in range(repeat):
            lev_vec = lev.lev_approx(x_mat)
            c_mat = lev.col_sample(x_mat, s_int2, lev_vec)[1]
            err2 += numpy.linalg.norm(xx_mat - numpy.dot(c_mat, c_mat.T), ord='fro') / xx_norm
        err2 /= repeat
        
        s_int3 = 1500
        err3 = 0
        for i in range(repeat):
            lev_vec = lev.lev_approx(x_mat)
            c_mat = lev.col_sample(x_mat, s_int3, lev_vec)[1]
            err3 += numpy.linalg.norm(xx_mat - numpy.dot(c_mat, c_mat.T), ord='fro') / xx_norm
        err3 /= repeat
        
        print('Approximation error for s=' + str(s_int1) + ':    ' + str(err1))
        print('Approximation error for s=' + str(s_int2) + ':    ' + str(err2))
        print('Approximation error for s=' + str(s_int3) + ':    ' + str(err3))
        self.assertTrue(err2 < err1)
        self.assertTrue(err3 < err2)
        
        
if __name__ == '__main__':
    unittest.main()