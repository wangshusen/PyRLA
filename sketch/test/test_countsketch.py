import numpy
import unittest
import sys

PyRLA_dir = '../../'
sys.path.append(PyRLA_dir)

import sketch.countsketch as cs

rawdata_mat = numpy.load(PyRLA_dir + 'data/YearPredictionMSD.npy', mmap_mode='r')
rawdata_mat = rawdata_mat[0:100000, :]
x_mat = rawdata_mat[:, 1:].T
m_int, n_int = x_mat.shape
y_mat = rawdata_mat[:, 0].reshape((1, n_int))
d_int = y_mat.shape[0]


xx_mat = numpy.dot(x_mat, x_mat.T)
xx_norm = numpy.linalg.norm(xx_mat, ord='fro')
xy_mat = numpy.dot(x_mat, y_mat.T)
xy_norm = numpy.linalg.norm(xy_mat, ord='fro')

class TestCountSketch(unittest.TestCase):
    def test_size(self):
        s_int = 19
        
        c_mat = cs.countsketch(x_mat, s_int)
        self.assertEqual(c_mat.shape[0], m_int)
        self.assertEqual(c_mat.shape[1], s_int)
        
        c_mat, d_mat = cs.countsketch2(x_mat, y_mat, s_int)
        self.assertEqual(c_mat.shape[0], m_int)
        self.assertEqual(c_mat.shape[1], s_int)
        self.assertEqual(d_mat.shape[0], d_int)
        self.assertEqual(d_mat.shape[1], s_int)
        
    def test_multiply_error(self):
        '''
        Test the function "countsketch"
        As the sketch size s_int increases, the approximation error should decrease.
        If the test fails, say twice in 10 tests, it is fine.
        '''
        repeat = 10
        
        s_int1 = 150
        err1 = 0
        for i in range(repeat):
            c_mat = cs.countsketch(x_mat, s_int1)
            err1 += numpy.linalg.norm(xx_mat - numpy.dot(c_mat, c_mat.T), ord='fro') / xx_norm
        err1 /= repeat
        
        s_int2 = 400
        err2 = 0
        for i in range(repeat):
            c_mat = cs.countsketch(x_mat, s_int2)
            err2 += numpy.linalg.norm(xx_mat - numpy.dot(c_mat, c_mat.T), ord='fro') / xx_norm
        err2 /= repeat
        
        s_int3 = 1500
        err3 = 0
        for i in range(repeat):
            c_mat = cs.countsketch(x_mat, s_int3)
            err3 += numpy.linalg.norm(xx_mat - numpy.dot(c_mat, c_mat.T), ord='fro') / xx_norm
        err3 /= repeat
        
        print('Approximation error for s=' + str(s_int1) + ':    ' + str(err1))
        print('Approximation error for s=' + str(s_int2) + ':    ' + str(err2))
        print('Approximation error for s=' + str(s_int3) + ':    ' + str(err3))
        self.assertTrue(err2 < err1)
        self.assertTrue(err3 < err2)
        
        
    def test_multiply_error2(self):
        '''
        Test the function "countsketch2"
        As the sketch size s_int increases, the approximation error should decrease.
        If the test fails, say twice in 10 tests, it is fine.
        '''
        
        repeat = 10
        
        s_int1 = 150
        err1xx = 0
        err1xy = 0
        for i in range(repeat):
            c_mat, d_mat = cs.countsketch2(x_mat, y_mat, s_int1)
            err1xx += numpy.linalg.norm(xx_mat - numpy.dot(c_mat, c_mat.T), ord='fro') / xx_norm
            err1xy += numpy.linalg.norm(xy_mat - numpy.dot(c_mat, d_mat.T), ord='fro') / xy_norm
        err1xx /= repeat
        err1xy /= repeat
        
        s_int2 = 400
        err2xx = 0
        err2xy = 0
        for i in range(repeat):
            c_mat, d_mat = cs.countsketch2(x_mat, y_mat, s_int2)
            err2xx += numpy.linalg.norm(xx_mat - numpy.dot(c_mat, c_mat.T), ord='fro') / xx_norm
            err2xy += numpy.linalg.norm(xy_mat - numpy.dot(c_mat, d_mat.T), ord='fro') / xy_norm
        err2xx /= repeat
        err2xy /= repeat
        
        s_int3 = 1500
        err3xx = 0
        err3xy = 0
        for i in range(repeat):
            c_mat, d_mat = cs.countsketch2(x_mat, y_mat, s_int3)
            err3xx += numpy.linalg.norm(xx_mat - numpy.dot(c_mat, c_mat.T), ord='fro') / xx_norm
            err3xy += numpy.linalg.norm(xy_mat - numpy.dot(c_mat, d_mat.T), ord='fro') / xy_norm
        err3xx /= repeat
        err3xy /= repeat
        
        print('Approximation error xx for s=' + str(s_int1) + ':    ' + str(err1xx))
        print('Approximation error xx for s=' + str(s_int2) + ':    ' + str(err2xx))
        print('Approximation error xx for s=' + str(s_int3) + ':    ' + str(err3xx))
        self.assertTrue(err2xx < err1xx)
        self.assertTrue(err3xx < err2xx)
        
        print('Approximation error xy for s=' + str(s_int1) + ':    ' + str(err1xy))
        print('Approximation error xy for s=' + str(s_int2) + ':    ' + str(err2xy))
        print('Approximation error xy for s=' + str(s_int3) + ':    ' + str(err3xy))
        self.assertTrue(err2xy < err1xy)
        self.assertTrue(err3xy < err2xy)
        
        
if __name__ == '__main__':
    unittest.main()