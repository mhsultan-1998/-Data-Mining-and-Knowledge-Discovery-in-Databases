import numpy as np
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 3: Ranking webpages using counts of in-links (15 points)
    In this problem, you will implement the ranking algorithm for webpages using the counts of in-links. 
 ------------------------------------ 
      Search Engine (Version 2) 
 ------------------------------------ 
 The earliest version of Google search is simply counting the number of in-links 
 (the hyperlinks from other webpages to this webpage) 
    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    Given an adjacency matrix A of a webpage hyperlink network, compute the counts of in-links for each webpage
    in the network. 
    ---- Inputs: --------
        * A: the adjacency matrix indicating the hyperlinks between webpages, a numpy matrix of shape (n X n).
          A[i][j] =1, if there is a hyperlink from webpage j to webpage i; A[j][i]=0, if there is no hyperlink 
          from webpage j to webpage i.
    ---- Outputs: --------
        * c: the number of in-links for each webpage, a numpy vector of length n, where c[i] is the number of 
          in-links pointing to webpage i.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def count_in_links(A):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    c = np.sum(A, axis=1)
    #########################################
    return c
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test3.py:test_count_in_links
	OR 
        nosetests -v test3.py:test_count_in_links
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    Given the counts of in-links of each webpage and the dataframe storing all the webpages (X1), insert a new
    column (named 'Inlinks') to the dataframe X1 to store the number of inlinks for each webpage. 
    ---- Inputs: --------
        * X1: a dataframe of all the webpages containing the column of word frequency, where each row 
          represents a webpage.
        * c: the number of in-links for each webpage, a numpy vector of length n, where c[i] is the number of 
          in-links pointing to webpage i.
    ---- Outputs: --------
        * X1: a dataframe of all the webpages containing the column of word frequency, where each row 
          represents a webpage.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def add_column_inlinks(X1, c):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    X1['Inlinks'] = c
    #########################################
    return X1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test3.py:test_add_column_inlinks
	OR 
        nosetests -v test3.py:test_add_column_inlinks
        ---------------------------------------------------
    '''
    
    ''' 
    If you have passed the above test, the result data should have been saved into a file, named "data_X2.csv",
    you could take a look at this file for the result.
    
    '''
    

#----------------------------------------------------
'''
    (Rank Webpages by Word Frequency) Given a dataframe (X2) of all the webpages, rank all the webpages by 
    descending order of # inlinks ('InLinks'). 
    ---- Inputs: --------
        * X2: a dataframe of all the webpages containing the columns of word frequency and number of inlinks,
          where each row represents a webpage.
    ---- Outputs: --------
        * R2: a dataframe of all the webpages, where the webpages are sorted according to descending order of 
          the number of in-links.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def rank_inlinks(X2):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    R2 = X2.sort_values(by='Inlinks', ascending=False)
    #########################################
    return R2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test3.py:test_rank_inlinks
	OR 
        nosetests -v test3.py:test_rank_inlinks
        ---------------------------------------------------
    '''
    
    ''' 
    If you have passed the above test, the result data should have saved into a file, named "data_R2.csv". You 
    could take a look at this file for the ranking result.
    (Demo) Now you can test the website that you have built by typing the following in the terminal: python3 
    demo2.py
    Then open the browser and view the URL showed in the terminal, for example: http://127.0.0.1:5000/
    
    '''
    


#--------------------------------------------

''' 
    TEST problem 3: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test3.py
	OR 
        nosetests -v test3.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 3 (15 points in total)--------------------- ... ok
        * (5 points) count_in_links ... ok
        * (5 points) add_column_inlinks ... ok
        * (5 points) rank_inlinks ... ok
        ----------------------------------------------------------------------
        Ran 3 tests in 0.736s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* A:  the adjacency matrix indicating the hyperlinks between webpages, a numpy matrix of shape (n X n). A[i][j]
 =1, if there is a hyperlink from webpage j to webpage i; A[j][i]=0, if there is no hyperlink from webpage j to
 webpage i. 
* c:  the number of in-links for each webpage, a numpy vector of length n, where c[i] is the number of in-links
  pointing to webpage i. 
* X1:  a dataframe of all the webpages containing the column of word frequency, where each row represents a 
  webpage. 
* X2:  a dataframe of all the webpages containing the columns of word frequency and number of inlinks, where 
  each row represents a webpage. 
* R2:  a dataframe of all the webpages, where the webpages are sorted according to descending order of the 
  number of in-links. 
* n:  the number of all webpages in the network, an integer scalar. 

'''
#--------------------------------------------