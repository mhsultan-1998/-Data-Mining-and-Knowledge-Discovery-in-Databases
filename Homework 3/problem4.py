import numpy as np
import pandas as pd
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 4: Ranking webpages using PageRank (AKA random walk with restart using matrix G) (48 points)
    In this problem, you will implement the ranking algorithm for webpages using PageRank 
    (random walk with restart on the revised hyperlink graph to solve both sink node and sink region problems).
 ------------------------------------ 
      Search Engine (Version 3) 
 ------------------------------------ 
 In this version of the search engine, we implement use random walk method on a revised hyperlink graph to 
 address both sink node and sink region problem. (1) If a webpage has no hyperlink (which is a sink node),
 add a hyperlink from this webpage to all the webpages in the network (including itself); (2) for every step 
 of the random walk, the user will have a small probability to jump to a random webpage ignoring the 
 hyperlinks and a large probability to follow the hyperlink
    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    Given an adjacency matrix A of a webpage hyperlink network, compute the count of out-links 
    (also called 'degree') for each webpage in the network. 
    ---- Inputs: --------
        * A: the adjacency matrix indicating the hyperlinks between webpages, a numpy matrix of shape (n X n).
          A[i,j] =1, if there is a hyperlink from webpage j to webpage i; A[i][j]=0, if there is no hyperlink 
          from webpage j to webpage i.
    ---- Outputs: --------
        * d: the vector storing the out-link degrees of all webpages, a numpy vector of length n. d[i] 
          represents the out-link degree of webpage i, which is the number of hyperlinks on webpage i.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def count_out_links(A):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    d = np.sum(A, axis=0)
    #########################################
    return d
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test4.py:test_count_out_links
	OR 
        nosetests -v test4.py:test_count_out_links
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given an adjacency matrix A (containing sink nodes), find all sink nodes in the network, then revise the 
    adjacency matrix A by adding a hyperlink from each sink node to all the webpages (including itself). 
    ---- Inputs: --------
        * A: the adjacency matrix indicating the hyperlinks between webpages, a numpy matrix of shape (n X n).
          A[i,j] =1, if there is a hyperlink from webpage j to webpage i; A[i][j]=0, if there is no hyperlink 
          from webpage j to webpage i. 
    ---- Outputs: --------
        * A: the adjacency matrix indicating the hyperlinks between webpages, a numpy matrix of shape (n X n).
          A[i,j] =1, if there is a hyperlink from webpage j to webpage i; A[i][j]=0, if there is no hyperlink 
          rom webpage j to webpage i.
    ---- Hints: --------
        * This problem can be solved using 3 line(s) of code.
'''
#---------------------
def remove_sink_nodes(A):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    #if all values in a column in the adjacency matrix sum to 0, fill that column with 1s.
    s, r, c, k, ze = list(np.sum(A, axis=0)), list(A.shape)[0], list(A.shape)[1], 0, [1]*list(A.shape)[0]
    while k < len(s):
        if s[k] == 0:
            A[:, k] = ze
        k += 1
    #########################################
    return A
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test4.py:test_remove_sink_nodes
	OR 
        nosetests -v test4.py:test_remove_sink_nodes
        ---------------------------------------------------
    '''
    
    ''' 
    If you have passed the above test, the result data should have been saved into a file, named "data_A_.csv".
    You could take a look at this file for the result.
    
    '''
    

#----------------------------------------------------
'''
    Given an adjacency matrix A_ of a webpage hyperlink network (after fixing all sink nodes), compute the 
    probability transition matrix S. 
    ---- Inputs: --------
        * A_: the adjacency matrix indicating the hyperlinks between webpages 
        (where all the sink nodes are fixed), a numpy matrix of shape (n X n). A_[i,j] =1, if there is a 
        hyperlink from webpage j to webpage i; A_[i][j]=0, if there is no hyperlink from webpage j to webpage 
        i.
    ---- Outputs: --------
        * S: the probability transition matrix among webpages (after solving the sink node problem), a (n by n)
          numpy matrix.  S[i,j] represents the probability of a random-walk user moving from webpage j to 
          webpage i by clicking a hyperlink on webpage j. The values in each column of matrix S should sum to
          1.
    ---- Hints: --------
        * You could use the functions you have implemented in the previous problems. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def compute_S(A_):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    S = A_.astype(float)
    s, k, k1, r, c = list(np.sum(A_, axis=0)), 0, 0, list(A_.shape)[0], list(A_.shape)[1]
    d = 0
    while d < r:
        t = list(S[d])
        k = 0
        while k < c:
            if t[k] == 1:
                t[k] = 1/(s[k])
            k += 1
        S[d] = t
        d += 1
    #########################################
    return S
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test4.py:test_compute_S
	OR 
        nosetests -v test4.py:test_compute_S
        ---------------------------------------------------
    '''
    
    ''' 
    If you have passed the above test, the result data should have been saved into a file, named "data_S.csv". 
    You could take a look at this file for the result.
    
    '''
    

#----------------------------------------------------
'''
    Given the probability transition matrix S of a webpage hyperlink network, fix the sink region problem in 
    the probability transition matrix by allowing the random walk to have a large probability (alpha) to follow
    the hyperlinks and a small probability (1-alpha) to restart 
    (jump to a random webpage, including itself, ignoring hyperlinks). 
    ---- Inputs: --------
        * S: the probability transition matrix among webpages (after solving the sink node problem), a (n by n)
          numpy matrix.  S[i,j] represents the probability of a random-walk user moving from webpage j to 
          webpage i by clicking a hyperlink on webpage j. The values in each column of matrix S should sum to 
          1.
        * alpha: a float scalar value between 0 and 1, which is the probability of following the hyperlinks in
          the random walk.
    ---- Outputs: --------
        * G: the probability transition matrix of the hyperlink network 
          (after fixing sink nodes and sink regions), a (n by n) numpy matrix of float values. G[i,j] 
          represents the probability of a user moving from webpage j to webpage i either by clicking a 
          hyperlink or randomly jumping to webpage j. The values in each column of matrix G should sum to 1.
    ---- Hints: --------
        * You could use the ones_like(X) function in numpy to create an all-one matrix of the same shape as 
          matrix X. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_G(S, alpha):
    #########################################
    ## INSERT YOUR CODE HERE (6 points)
    z = np.full((list(S.shape)[0],list(S.shape)[1]), 1/(list(S.shape)[0]))
    G = np.add((np.multiply(alpha, S)), (np.multiply((1-alpha), z)))
    #########################################
    return G
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test4.py:test_compute_G
	OR 
        nosetests -v test4.py:test_compute_G
        ---------------------------------------------------
    '''
    
    ''' 
    If you have passed the above test, the result data should have been saved into a file, named "data_G.csv". 
    You could take a look at this file for the result.
    
    '''
    

#----------------------------------------------------
'''
    (One-step Random Walk) Given the probability transition matrix G of a webpage hyperlink network and the 
    vector x of the current scores for all webpages, compute the new scores of all webpages after one step of 
    random walk and return these scores as vector y. 
    ---- Inputs: --------
        * G: the probability transition matrix of the hyperlink network
          (after fixing sink nodes and sink regions), a (n by n) numpy matrix of float values.  G[i,j]
          represents the probability of a user moving from webpage j to webpage i either by clicking a 
          hyperlink or randomly jumping to webpage j. The values in each column of matrix G should sum to 1.
        * x: the vector of probability distribution of the random walker on all webpages in the network, a 
          numpy vector of length n. x[i] represents the probability of the random walker in Webpage i.
    ---- Outputs: --------
        * y: .
    ---- Hints: --------
        * You could use matrix-vector multiplication to compute the result of one-step random walk. 
        * To multiply a numpy matrix A with a numpy vector b, you can use A@b. 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def random_walk_one_step(G, x):
    #########################################
    ## INSERT YOUR CODE HERE (6 points)
    y = np.dot(G, x)
    #########################################
    return y
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test4.py:test_random_walk_one_step
	OR 
        nosetests -v test4.py:test_random_walk_one_step
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given two vectors of the webpage scores, check whether or not all the values in the two vectors are close
    enough. For example, if the tolerance (tol) is 0.1, the two values a and b are called 'close' if the 
    absolute difference between the two values |a-b|<=0.1. Now when we have two vectors of values x, y, the 
    two vectors are 'close' if all values of the two vectors are 'close'. 
    ---- Inputs: --------
        * x: the vector of probability distribution of the random walker on all webpages in the network, a 
          numpy vector of length n. x[i] represents the probability of the random walker in Webpage i.
        * y: .
        * tol: the tolerance level, a float scalar, if all elements in two vectors x and y are close 
          (|x[i]-y[i]|<= tol), then the x and y are close to each other.
    ---- Outputs: --------
        * c: .
    --------------------------------------
    ---- Example: --------
        For example, suppose we have two vectors x=[1.05, 4.03], y=[1.02, 4.05], if the tolerance is 0.1, then
        x and y are 'close'. In this case, the returned result (c) should be True
        For example, suppose we have two vectors m=[1.05, 4.03, 0.16], n=[1.02, 4.05, 0.35], if the tolerance 
        is 0.1, then m and n are NOT 'close', because the last element is not close to each other within the
        tolerance. In this case, the returned result (c) should be False 
    --------------------------------------
    ---- Hints: --------
        * numpy has a function 'allclose()' to test whether two vectors are close enough, by setting the 
          absolute tolerance parameter (atol). 
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def all_close(x, y, tol=0.01):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    for x in np.absolute(np.subtract(x,y)):
        if x > tol:
            return False
        else:
            c = True
    #########################################
    return c
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test4.py:test_all_close
	OR 
        nosetests -v test4.py:test_all_close
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given the probability transition matrix G of a webpage network and a vector of the initial probability 
    distribution (x) on all the webpages, compute the stationary probability distribution on the webpages 
    (also called 'pagerank scores') and return them as vector x. 
	 (1) use a for-loop to repeat the random walk for max_steps times. 
	 (2) for each step inside the for-loop, 
		  (2.1) update the webpage scores using one-step random walk 
		  (2.2) check whether the scores before the random walk (x) and the scores after the random walk (y) 
          are close enough within the tolerance (tol) 
		  (2.3) if the two score vectors are close, break the for-loop, otherwise update the score vector x
          with the new vector y. 
    ---- Inputs: --------
        * G: the probability transition matrix of the hyperlink network 
          (after fixing sink nodes and sink regions), a (n by n) numpy matrix of float values.  G[i,j] 
          represents the probability of a user moving from webpage j to webpage i either by clicking a 
          hyperlink
        or randomly jumping to webpage j. The values in each column of matrix G should sum to 1.
        * x: the vector of probability distribution of the random walker on all webpages in the network, a 
          numpy vector of length n. x[i] represents the probability of the random walker in Webpage i.
        * tol: the tolerance level, a float scalar, if all elements in two vectors x and y are close
          (|x[i]-y[i]|<= tol), then the x and y are close to each other.
        * max_steps: the maximum number of random walk steps, an integer scalar.
    ---- Outputs: --------
        * x: the vector of probability distribution of the random walker on all webpages in the network, a 
          numpy vector of length n. x[i] represents the probability of the random walker in Webpage i.
    ---- Hints: --------
        * You could use the 'break' statement in python to break a for-loop. 
        * This function will take less than 1 second to run, if this function runs for a very long time 
          without stopping, it means that the program never encountered the 'break' statement inside the 
          for-loop. 
        * This problem can be solved using 5 line(s) of code.
'''
#---------------------
def random_walk(G, x, tol=0.01, max_steps=100):
    #########################################
    ## INSERT YOUR CODE HERE (6 points)
    z = 0
    while z < max_steps:
        y = random_walk_one_step(G, x)
        if all_close(x, y, tol) == True:
            break
        else:
            x = y
        z += 1
    #########################################
    return x
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test4.py:test_random_walk
	OR 
        nosetests -v test4.py:test_random_walk
        ---------------------------------------------------
    '''
    
    ''' 
    If you have passed the above test, the result data should have been saved into a file, named "data_x.csv".
    You could take a look at this file for the result.
    
    '''
    

#----------------------------------------------------
'''
    Given the pagerank scores of each webpage in (x) and the dataframe storing all the webpage data (X2), add 
    a new column called 'PageRank' to dataframe (X2) to store the PageRank scores for all webpages. 
    ---- Inputs: --------
        * X2: a dataframe of all the webpages containing the columns of word frequency and number of inlinks,
          where each row represents a webpage.
        * x: the vector of probability distribution of the random walker on all webpages in the network, a 
          numpy vector of length n. x[i] represents the probability of the random walker in Webpage i.
    ---- Outputs: --------
        * X2: a dataframe of all the webpages containing the columns of word frequency and number of inlinks, 
          where each row represents a webpage.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def add_column_pagerank(X2, x):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    X2['PageRank'] = x
    #########################################
    return X2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test4.py:test_add_column_pagerank
	OR 
        nosetests -v test4.py:test_add_column_pagerank
        ---------------------------------------------------
    '''
    
    ''' 
    If you have passed the above test, the result data should have been saved into a file, named "data_X3.csv",
    you could take a look at this file for the result.
    
    '''
    

#----------------------------------------------------
'''
    (Rank Webpages by PageRank Score) Given a dataframe (X3) of all the webpages, rank all the pages based
    upon descending order of pagerank scores (PageRank). 
    ---- Inputs: --------
        * X3: a dataframe of all the webpages containing the columns of pagerank socres, word frequency and
          number of inlinks, where each row represents a webpage.
    ---- Outputs: --------
        * R3: a dataframe of all the webpages, where the webpages are sorted according to descending order of 
          the pagerank scores.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def rank_pages(X3):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    R3 = X3.sort_values(by='PageRank', ascending=False)
    #########################################
    return R3
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test4.py:test_rank_pages
	OR 
        nosetests -v test4.py:test_rank_pages
        ---------------------------------------------------
    '''
    
    ''' 
    If you have passed the above test, the result data should have been saved into a file, named "data_R3.csv".
    You could take a look at this file for the ranking result.
    (Demo) Now you can test the website that you have built by typing the following in the terminal: python3 
    demo3.py
    Then open the browser and view the URL showed in the terminal, for example: http://127.0.0.1:5000/
    
    '''
    


#--------------------------------------------

''' 
    TEST problem 4: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test4.py
	OR 
        nosetests -v test4.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 4 (48 points in total)--------------------- ... ok
        * (5 points) count_out_links ... ok
        * (5 points) remove_sink_nodes ... ok
        * (5 points) compute_S ... ok
        * (6 points) compute_G ... ok
        * (6 points) random_walk_one_step ... ok
        * (5 points) all_close ... ok
        * (6 points) random_walk ... ok
        * (5 points) add_column_pagerank ... ok
        * (5 points) rank_pages ... ok
        ----------------------------------------------------------------------
        Ran 9 tests in 0.006s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* A:  the adjacency matrix indicating the hyperlinks between webpages, a numpy matrix of shape (n X n). 
  A[i,j] = 1, if there is a hyperlink from webpage j to webpage i; A[i][j]=0, if there is no hyperlink from 
  webpage j to webpage i. 
* d:  the vector storing the out-link degrees of all webpages, a numpy vector of length n. d[i] represents the
  out-link degree of webpage i, which is the number of hyperlinks on webpage i. 
* A_:  the adjacency matrix indicating the hyperlinks between webpages (where all the sink nodes are fixed), a 
  numpy matrix of shape (n X n). A_[i,j] =1, if there is a hyperlink from webpage j to webpage i; A_[i][j]=0, 
  if there is no hyperlink from webpage j to webpage i. 
* S:  the probability transition matrix among webpages (after solving the sink node problem), a (n by n) numpy
  matrix.  S[i,j] represents the probability of a random-walk user moving from webpage j to webpage i by 
  clicking a hyperlink on webpage j. The values in each column of matrix S should sum to 1. 
* G:  the probability transition matrix of the hyperlink network (after fixing sink nodes and sink regions), a
  (n by n) numpy matrix of float values.  G[i,j] represents the probability of a user moving from webpage j to 
  webpage i either by clicking a hyperlink or randomly jumping to webpage j. The values in each column of 
  matrix G should sum to 1. 
* alpha:  a float scalar value between 0 and 1, which is the probability of following the hyperlinks in the 
  random walk. 
* n:  the number of all webpages in the network, an integer scalar. 
* x:  the vector of probability distribution of the random walker on all webpages in the network, a numpy 
  vector of length n. x[i] represents the probability of the random walker in Webpage i. 
* tol:  the tolerance level, a float scalar, if all elements in two vectors x and y are close 
  (|x[i]-y[i]|<= tol), then the x and y are close to each other. 
* max_steps:  the maximum number of random walk steps, an integer scalar. 
* X2:  a dataframe of all the webpages containing the columns of word frequency and number of inlinks, where 
  each row represents a webpage. 
* X3:  a dataframe of all the webpages containing the columns of pagerank socres, word frequency and number of 
  inlinks, where each row represents a webpage. 
* R3:  a dataframe of all the webpages, where the webpages are sorted according to descending order of the 
  pagerank scores. 

'''
#--------------------------------------------