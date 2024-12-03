import numpy as np
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 1: Getting Familiar with NumPy Package (Part 2) (30 points)
    In this problem, we will continue to get familiar with numpy package. Specially, we will learn about the 
    matrix operations in linear algebra

    A list of all variables being used in this problem is provided at the end of this file.
'''

#--------------------------
def Terms_and_Conditions():
    ''' 
        By submitting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, 
           sending your code segment to another student, putting your solution online or lending your laptop 
           (if your laptop contains your solution or your Dropbox automatically copied your solution from your
           desktop computer and your laptop) to another student to work on this homework will violate this 
           term.
       (2) Not using anyone's code in this homework and building your own solution. For example, using some 
           code segments from another student or online resources due to any reason (like too busy recently)
           will violate this term. Changing other's code as your solution (such as changing the variable names)
           will also violate this term.
       (3) When discussing with any other students about this homework, only discuss high-level ideas or use
           pseudo-code. Don't discuss about the solution at the code level. For example, two students discuss 
           about the solution of a function (which needs 5 lines of code to solve) and they then work on the 
           solution "independently", however the code of the two solutions are exactly the same, or only with
           minor differences (variable names are different). In this case, the two students violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the WPI Academic Honesty Policy. 
      For more details, please visit: https://www.wpi.edu/about/policies/academic-integrity/dishonesty
      Note: we may use the Stanford Moss system to check your code for code similarity.
      https://theory.stanford.edu/~aiken/moss/
      Historical Data: in one year, we ended up finding 25% of the students in that class violating this term 
      in their homework submissions and we handled ALL of these violations according to the WPI Academic 
      Honesty Policy. 
    '''
    #*******************************************
    # CHANGE HERE: if you have read and agree with the term above, change "False" to "True".
    Read_and_Agree = True
    #*******************************************
    return Read_and_Agree

#----------------------------------------------------
'''
     Given an input numpy matrix X, compute the transpose of the matrix X. For example X is 
	 1,2,3 
	 4,5,6 
	 Then the transpose of matrix X is 
	 1,4 
	 2,5 
	 3,6. 
    ---- Inputs: --------
        * X: a numpy matrix (2D array).
    ---- Outputs: --------
        * Xt: a numpy matrix (2D array), which is the transpose of matrix X.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def transpose(X):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)  list(A.shape)[0], list(A.shape)[1]
    Xt = X.reshape((X.shape)[1], (X.shape)[0])
    #########################################
    return Xt
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_transpose
	OR 
        nosetests -v test1.py:test_transpose
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Create an identity matrix of shape n X n, all the values in the matrix should be 0, except the diagonal 
    elements (which should be 1). For example, an identity matrix of shape 3 X 3 will be 
	 1, 0, 0 
	 0, 1, 0 
	 0, 0, 1. 
    ---- Inputs: --------
        * n: an integer, the number of rows/columns in an identity matrix I.
    ---- Outputs: --------
        * I: a numpy identity matrix of shape n X n, where all diagonal elements are 1s and off-diagonal 
          elements are 0s.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def identity_matrix(n):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    I = np.identity(n)
    #########################################
    return I
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_identity_matrix
	OR 
        nosetests -v test1.py:test_identity_matrix
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given two matrices X (of shape r X k) and Y (k X c), the compute the matrix multiplication between the 
    matrix X and Y: Z = X Y 
 For example, if matrix X is: 
	 1, 2, 3 
	 4, 5, 6 
 and matrix Y is, 
	 -1, -4 
	 -2, -5 
	 -3, -6 
 then, ( Z = X Y) should be: 
	 -14, -32 
	 -32, -77 
	Here is how we calculated Z: 
	Z[0,0] =  1*(-1)+ 2*(-2) + 3*(-3) = -14 
	Z[0,1] =  1*(-4)+ 2*(-5) + 3*(-6) = -32 
	Z[1,0] =  4*(-1)+ 5*(-2) + 6*(-3) = -32 
	Z[1,1] =  4*(-4)+ 5*(-5) + 6*(-6) = -77 . 
    ---- Inputs: --------
        * X: a numpy matrix (2D array).
        * Y: a numpy matrix (2D array).
    ---- Outputs: --------
        * Z: a numpy matrix (2D array), which is the result of the matrix multiplication of matrices X and Y.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def matrix_multiplication(X, Y):
    #########################################
    ## INSERT YOUR CODE HERE (6 points)
    Z = np.dot(X, Y)
    #########################################
    return Z
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_matrix_multiplication
	OR 
        nosetests -v test1.py:test_matrix_multiplication
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given a matrix X (of shape r X c), the compute the inverse matrix of X (called Xi), so that  X Xi = I
    (identity matrix) 
 For example, if matrix X is: 
	 2, 0 
	 0, 4 
 the inverse matrix Xi of X is, 
	 1/2,  0 
	  0 , 1/4 
 So, when we compute the matrix multiplication X Xi, we get an identity matrix I: 
	  1, 0 
	  0, 1 
 For example, if matrix X is: 
	 1, 2 
	 3, 4 
 the inverse matrix Xi of X is, 
	  -2,  1 
	 3/2,-1/2 
 So, when we compute the matrix multiplication (X * Xi = I), we get an identity matrix I: 
	  1, 0 
	  0, 1 . 
    ---- Inputs: --------
        * X: a numpy matrix (2D array).
    ---- Outputs: --------
        * Xi: a numpy matrix (2D array), which is the inverse of matrix X.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def matrix_inverse(X):
    #########################################
    ## INSERT YOUR CODE HERE (6 points)
    Xi = np.linalg.inv(X) 
    #########################################
    return Xi
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_matrix_inverse
	OR 
        nosetests -v test1.py:test_matrix_inverse
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
     Given an input numpy array x (with missing values), compute a boolean array indicating the locations of 
     the missing values. 
	 For example, x = [1,2,?,3,?] 
	 Here ? represents a missing value (np.nan) 
	 Then output m should be 
	 m= [False, False, True, False, True] 
	 Which means that the 3rd and 5th elements are missing (True), while all the other elements are not 
     missing (False). 
    ---- Inputs: --------
        * x: a numpy vector (with missing values), x[i] = np.nan if the i-th element of x is missing.
    ---- Outputs: --------
        * m: a numpy boolean vector, where each element is True/False.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def is_missing(x):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    m = np.isnan(x)
    #########################################
    return m
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_is_missing
	OR 
        nosetests -v test1.py:test_is_missing
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
     Given an input boolean array (m), compute the inverse of the boolean array (m_). 
	 For example, if m= [False, False, True, False, True], 
	 The inverse of m will be: 
	 m_ = [True, True, False, True, False]. 
    ---- Inputs: --------
        * m: a numpy boolean vector, where each element is True/False.
    ---- Outputs: --------
        * m_: a numpy boolean vector, which is the inverse of vector m, m_[i] = False if m[i]=True.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def inverse(m):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    m_ = np.where(m == False, True, False) 
    #########################################
    return m_
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_inverse
	OR 
        nosetests -v test1.py:test_inverse
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
     Given an input numpy vector x and a boolean vector (m) of the same length as x, extract the subset of 
     elements in x indexed by m 
     (where the elements to be used are True in m, the elements to be discarded are False in m). 
	 For example, x = [1,2,3,4,5] and m = [True, True, False, True, False] 
	 Then the subset of the vector x indexed by m will be 
	 x_m= [1,2,4] 
	 For example, x = [1,2,3,4,5] and m = [True, False, False, False, True] 
	 Then the subset of the vector x indexed by m will be 
	 x_m= [1,5]. 
    ---- Inputs: --------
        * x: a numpy vector (with missing values), x[i] = np.nan if the i-th element of x is missing.
        * m: a numpy boolean vector, where each element is True/False.
    ---- Outputs: --------
        * x_m: a numpy vector, which is a subset of vector x indexed by m.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def subset_vector(x, m):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    x_m = x[m]
    #########################################
    return x_m
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_subset_vector
	OR 
        nosetests -v test1.py:test_subset_vector
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
     Given an input numpy matrix X and a boolean vector (m), extract the subset matrix in X indexed by m 
     (where the rows of X to be used are True in m, the rows of X to be discarded are False in m). 
	 For example, X = [[1,2], 
	                   [3,4], 
	                   [5,6]] 
	 The index vector m = [True, False, True], then the sub-matrix will be 
     (keeping the 1st and 3rd rows of the matrix X) 
	 X_m= [[1,2], 
	       [5,6]] 
	 If the index vector m = [False, True, True], then the sub-matrix will be 
     (keeping the 2nd and 3rd rows of the matrix X) 
	 X_m= [[3,4], 
	       [5,6]] . 
    ---- Inputs: --------
        * X: a numpy matrix (2D array).
        * m: a numpy boolean vector, where each element is True/False.
    ---- Outputs: --------
        * X_m: a numpy matrix, which is a subset of matrix X indexed by m.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def subset_matrix(X, m):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    X_m = X[m]
    #########################################
    return X_m
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_subset_matrix
	OR 
        nosetests -v test1.py:test_subset_matrix
        ---------------------------------------------------
    '''
    
    


#--------------------------------------------

''' 
    TEST problem 1: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py
	OR 
        nosetests -v test1.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 1 (30 points in total)--------------------- ... ok
        * (3 points) transpose ... ok
        * (3 points) identity_matrix ... ok
        * (6 points) matrix_multiplication ... ok
        * (6 points) matrix_inverse ... ok
        * (3 points) is_missing ... ok
        * (3 points) inverse ... ok
        * (3 points) subset_vector ... ok
        * (3 points) subset_matrix ... ok
        ----------------------------------------------------------------------
        Ran 8 tests in 0.006s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* X:  a numpy matrix (2D array). 
* Y:  a numpy matrix (2D array). 
* Z:  a numpy matrix (2D array), which is the result of the matrix multiplication of matrices X and Y. 
* Xt:  a numpy matrix (2D array), which is the transpose of matrix X. 
* Xi:  a numpy matrix (2D array), which is the inverse of matrix X. 
* r:  an integer, the number of rows in the matrix X. 
* c:  an integer, the number of columns in the matrix. 
* I:  a numpy identity matrix of shape n X n, where all diagonal elements are 1s and off-diagonal elements are 
      0s. 
* n:  an integer, the number of rows/columns in an identity matrix I. 
* k:  an integer, the number of columns in the matrix X. 
* x:  a numpy vector (with missing values), x[i] = np.nan if the i-th element of x is missing. 
* m:  a numpy boolean vector, where each element is True/False. 
* x_m:  a numpy vector, which is a subset of vector x indexed by m. 
* m_:  a numpy boolean vector, which is the inverse of vector m, m_[i] = False if m[i]=True. 
* X_m:  a numpy matrix, which is a subset of matrix X indexed by m. 

'''
#--------------------------------------------