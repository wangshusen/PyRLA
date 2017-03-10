import numpy

def countsketch(a_mat, s_int):
    '''
    Count Sketch for Dense Matrix
    
    Input
        a_mat: m-by-n dense matrix A;
        s_int: sketch size.
    Output
        sketch_a_mat: m-by-s matrix A * S.
        Here S is n-by-s sketching matrix.
    '''
    m_int, n_int = a_mat.shape
    hash_vec = numpy.random.choice(s_int, n_int, replace=True)
    sign_vec = numpy.random.choice(2, n_int, replace=True) * 2 - 1
    sketch_a_mat = numpy.zeros((m_int, s_int))
    for j in range(n_int):
        h = hash_vec[j]
        g = sign_vec[j]
        sketch_a_mat[:, h] += g * a_mat[:, j]
    return sketch_a_mat
    

def countsketch2(a_mat, b_mat, s_int):
    '''
    Count Sketch for 2 Dense Matrices
    
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
    hash_vec = numpy.random.choice(s_int, n_int, replace=True)
    sign_vec = numpy.random.choice(2, n_int, replace=True) * 2 - 1
    
    sketch_a_mat = numpy.zeros((m_int, s_int))
    for j in range(n_int):
        h = hash_vec[j]
        g = sign_vec[j]
        sketch_a_mat[:, h] += g * a_mat[:, j]
    
    sketch_b_mat = numpy.zeros((d_int, s_int))
    for j in range(n_int):
        h = hash_vec[j]
        g = sign_vec[j]
        sketch_b_mat[:, h] += g * b_mat[:, j]
        
    return sketch_a_mat, sketch_b_mat
    

    
    
    
