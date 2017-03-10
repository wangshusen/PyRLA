import numpy

def gaussian_proj(a_mat, s_int):
    '''
    Random Gaussian Projection
    
    Input
        a_mat: m-by-n dense matrix A;
        s_int: sketch size.
    Output
        sketch_a_mat: m-by-s matrix A * S.
        Here S is n-by-s sketching matrix.
    '''
    m_int, n_int = a_mat.shape
    s_mat = numpy.random.randn(n_int, s_int) / numpy.sqrt(s_int)
    sketch_a_mat = numpy.dot(a_mat, s_mat)
    return sketch_a_mat
    

def gaussian_proj2(a_mat, b_mat, s_int):
    '''
    Random Gaussian Projection for 2 Matrices
    
    Input
        a_mat: m-by-n dense matrix A;
        b_mat: d-by-n dense matrix B;
        s_int: sketch size.
    Output
        sketch_a_mat: m-by-s matrix A * S;
        sketch_b_mat: d-by-s matrix B * S.
        Here S is n-by-s sketching matrix
    '''
    m_int, n_int = a_mat.shape
    d_int = b_mat.shape[0]
    s_mat = numpy.random.randn(n_int, s_int) / numpy.sqrt(s_int)
    sketch_a_mat = numpy.dot(a_mat, s_mat)
    sketch_b_mat = numpy.dot(b_mat, s_mat)
    return sketch_a_mat, sketch_b_mat
    

    
    
    
