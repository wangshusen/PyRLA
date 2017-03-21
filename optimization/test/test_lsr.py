import numpy
import unittest
import sys

PyRLA_dir = '../../'
sys.path.append(PyRLA_dir)

from optimization import lsr


def approx_lsr(sketch_size, sketch_type):
    w_sketch_vec, obj_val = lsr.sketched_lsr(x_mat, y_vec, sketch_size=sketch_size, sketch_type=sketch_type)
    dist = w_sketch_vec - w_opt_vec
    dist = numpy.sum(dist ** 2)
    return obj_val, dist

class TestLSR(unittest.TestCase):
    def test_srft(self):
        print('######## SRFT ########')
        
        sketch_size = 3
        obj_val1, dist1 = approx_lsr(sketch_size, 'srft')
        print('Objective function value (s=' + str(sketch_size) + 'd): ' + str(obj_val1))
        print('Distance to optimal (s=' + str(sketch_size) + 'd): ' + str(dist1))
        
        sketch_size = 5
        obj_val2, dist2 = approx_lsr(sketch_size, 'srft')
        print('Objective function value (s=' + str(sketch_size) + 'd): ' + str(obj_val2))
        print('Distance to optimal (s=' + str(sketch_size) + 'd): ' + str(dist2))
        
        sketch_size = 10
        obj_val3, dist3 = approx_lsr(sketch_size, 'srft')
        print('Objective function value (s=' + str(sketch_size) + 'd): ' + str(obj_val3))
        print('Distance to optimal (s=' + str(sketch_size) + 'd): ' + str(dist3))
        
        self.assertTrue(dist3 < dist1)
        
    def test_count(self):
        print('######## Count Sketch ########')
        
        sketch_size = 3
        obj_val1, dist1 = approx_lsr(sketch_size, 'count')
        print('Objective function value (s=' + str(sketch_size) + 'd): ' + str(obj_val1))
        print('Distance to optimal (s=' + str(sketch_size) + 'd): ' + str(dist1))
        
        sketch_size = 5
        obj_val2, dist2 = approx_lsr(sketch_size, 'count')
        print('Objective function value (s=' + str(sketch_size) + 'd): ' + str(obj_val2))
        print('Distance to optimal (s=' + str(sketch_size) + 'd): ' + str(dist2))
        
        sketch_size = 10
        obj_val3, dist3 = approx_lsr(sketch_size, 'count')
        print('Objective function value (s=' + str(sketch_size) + 'd): ' + str(obj_val3))
        print('Distance to optimal (s=' + str(sketch_size) + 'd): ' + str(dist3))
        
        self.assertTrue(dist3 < dist1)
        
    def test_leverage(self):
        print('######## Leverage Score Sampling ########')
        
        sketch_size = 3
        obj_val1, dist1 = approx_lsr(sketch_size, 'leverage')
        print('Objective function value (s=' + str(sketch_size) + 'd): ' + str(obj_val1))
        print('Distance to optimal (s=' + str(sketch_size) + 'd): ' + str(dist1))
        
        sketch_size = 5
        obj_val2, dist2 = approx_lsr(sketch_size, 'leverage')
        print('Objective function value (s=' + str(sketch_size) + 'd): ' + str(obj_val2))
        print('Distance to optimal (s=' + str(sketch_size) + 'd): ' + str(dist2))
        
        sketch_size = 10
        obj_val3, dist3 = approx_lsr(sketch_size, 'leverage')
        print('Objective function value (s=' + str(sketch_size) + 'd): ' + str(obj_val3))
        print('Distance to optimal (s=' + str(sketch_size) + 'd): ' + str(dist3))
        
        self.assertTrue(dist3 < dist1)
        
    def test_shrink(self):
        print('######## Shrinked Leverage Score Sampling ########')
        
        sketch_size = 3
        obj_val1, dist1 = approx_lsr(sketch_size, 'shrink')
        print('Objective function value (s=' + str(sketch_size) + 'd): ' + str(obj_val1))
        print('Distance to optimal (s=' + str(sketch_size) + 'd): ' + str(dist1))
        
        sketch_size = 5
        obj_val2, dist2 = approx_lsr(sketch_size, 'shrink')
        print('Objective function value (s=' + str(sketch_size) + 'd): ' + str(obj_val2))
        print('Distance to optimal (s=' + str(sketch_size) + 'd): ' + str(dist2))
        
        sketch_size = 10
        obj_val3, dist3 = approx_lsr(sketch_size, 'shrink')
        print('Objective function value (s=' + str(sketch_size) + 'd): ' + str(obj_val3))
        print('Distance to optimal (s=' + str(sketch_size) + 'd): ' + str(dist3))
        
        self.assertTrue(dist3 < dist1)
        
if __name__ == '__main__':
    rawdata_mat = numpy.load(PyRLA_dir + 'data/YearPredictionMSD.npy', mmap_mode='r')
    rawdata_mat = rawdata_mat[0:50000, :]
    x_mat = rawdata_mat[:, 1:]
    n_int, d_int = x_mat.shape
    y_vec = rawdata_mat[:, 0].reshape((n_int, 1))

    w_opt_vec = numpy.dot(numpy.linalg.pinv(x_mat), y_vec)
    residual = numpy.dot(x_mat, w_opt_vec) - y_vec
    opt_obj_val = numpy.sum(residual ** 2) / n_int
    print('Optimal objective value is ' + str(opt_obj_val))
    
    unittest.main()
    