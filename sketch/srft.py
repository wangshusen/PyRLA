import numpy

# Remark:
#   Real FFT with even n is faster than real FFT with odd n.
#   I do not know why.

def realfft_col(a_mat):
    '''
    Real Fast Fourier Transform (FFT) Independently Applied to Each Column of A
    
    Input
        a_mat: n-by-d dense NumPy matrix.
    
    Output
        c_mat: n-by-d matrix C = F * A.
        Here F is the n-by-n orthogonal real FFT matrix (not explicitly formed)
    
    Notice that $C^T * C = A^T * A$; 
    however, $C * C^T = A * A^T$ is not true.
    '''
    n_int = a_mat.shape[0]
    fft_mat = numpy.fft.fft(a_mat, n=None, axis=0) / numpy.sqrt(n_int)
    if n_int % 2 == 1:
        cutoff_int = int((n_int+1) / 2)
        idx_real_vec = list(range(1, cutoff_int))
        idx_imag_vec = list(range(cutoff_int, n_int))
    else:
        cutoff_int = int(n_int/2)
        idx_real_vec = list(range(1, cutoff_int))
        idx_imag_vec = list(range(cutoff_int+1, n_int))
    c_mat = fft_mat.real
    c_mat[idx_real_vec, :] *= numpy.sqrt(2)
    c_mat[idx_imag_vec, :] = fft_mat[idx_imag_vec, :].imag * numpy.sqrt(2)
    return c_mat


def realfft_row(a_mat):
    '''
    Real Fast Fourier Transform (FFT) Independently Applied to Each Row of A
    
    Input
        a_mat: m-by-n dense NumPy matrix.
    
    Output
        c_mat: m-by-n matrix C = A * F.
        Here F is the n-by-n orthogonal real FFT matrix (not explicitly formed)
    
    Notice that $C * C^T = A * A^T$; 
    however, $C^T * C = A^T * A$ is not true.
    '''
    n_int = a_mat.shape[1]
    fft_mat = numpy.fft.fft(a_mat, n=None, axis=1) / numpy.sqrt(n_int)
    if n_int % 2 == 1:
        cutoff_int = int((n_int+1) / 2)
        idx_real_vec = list(range(1, cutoff_int))
        idx_imag_vec = list(range(cutoff_int, n_int))
    else:
        cutoff_int = int(n_int/2)
        idx_real_vec = list(range(1, cutoff_int))
        idx_imag_vec = list(range(cutoff_int+1, n_int))
    c_mat = fft_mat.real
    c_mat[:, idx_real_vec] *= numpy.sqrt(2)
    c_mat[:, idx_imag_vec] = fft_mat[:, idx_imag_vec].imag * numpy.sqrt(2)
    return c_mat

    
def srft(a_mat, s_int):
    '''
    Subsampled Randomized Fourier Transform (SRFT) for Dense Matrix
    
    Input
        a_mat: m-by-n dense NumPy matrix;
        s_int: sketch size.
    
    Output
        c_mat: m-by-s sketch C = A * S.
        Here S is the sketching matrix (not explicitly formed)
    '''
    n_int = a_mat.shape[1]
    sign_vec = numpy.random.choice(2, n_int) * 2 - 1
    idx_vec = numpy.random.choice(n_int, s_int, replace=False)
    a_mat = a_mat * sign_vec.reshape(1, n_int)
    a_mat = realfft_row(a_mat)
    c_mat = a_mat[:, idx_vec] * numpy.sqrt(n_int / s_int)
    return c_mat

    
def srft2(a_mat, b_mat, s_int):
    '''
    Subsampled Randomized Fourier Transform (SRFT) for Dense Matrix
    
    Input
        a_mat: m-by-n dense NumPy matrix;
        b_mat: d-by-n dense NumPy matrix;
        s_int: sketch size.
    
    Output
        c_mat: m-by-s sketch C = A * S;
        d_mat: d-by-s sketch D = B * S.
        Here S is the sketching matrix (not explicitly formed)
    '''
    n_int = a_mat.shape[1]
    sign_vec = numpy.random.choice(2, n_int) * 2 - 1
    idx_vec = numpy.random.choice(n_int, s_int, replace=False)
    
    a_mat = a_mat * sign_vec.reshape(1, n_int)
    a_mat = realfft_row(a_mat)
    c_mat = a_mat[:, idx_vec] * numpy.sqrt(n_int / s_int)
    
    b_mat = b_mat * sign_vec.reshape(1, n_int)
    b_mat = realfft_row(b_mat)
    d_mat = b_mat[:, idx_vec] * numpy.sqrt(n_int / s_int)
    return c_mat, d_mat
