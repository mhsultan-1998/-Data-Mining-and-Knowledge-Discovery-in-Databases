import re
import pandas as pd
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 1: Ranking Webpages using Counts of Keywords (15 points)
    In this problem, we will implement a webpage ranking algorithm: We will use the frequency of keywords in 
    the webpages to determine the ranking. We will use this method on a small dataset of 8 webpages and create
    a demo to show the result of the search engine. 
 ------------------------------------------------------ 
      Search Engine (Version 1): Rank by Word Frequency 
 ------------------------------------------------------ 
 In this version of the search engine, the webpages with the most occurrence of the searched keyword will 
 appear on the top of the result
    A list of all variables being used in this problem is provided at the end of this file.
'''

#--------------------------
def Terms_and_Conditions():
    ''' 
        By submitting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, 
       sending your code segment to another student, putting your solution online or lending your laptop 
       (if your laptop contains your solution or your Dropbox automatically copied your solution from your
        desktop computer and your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework and building your own solution. For example, using some 
       code segments from another student or online resources due to any reason (like too busy recently) will 
       violate this term. Changing other's code as your solution (such as changing the variable names) will 
       also violate this term.
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
    (Load Webpage Data) Given a file name of a CSV file containing all the webpages, load the CSV file into a 
    pandas dataframe. Each row contains information from a webpage. The data frame has 4 columns: ID, URL, 
    Title and Description. 
    ---- Inputs: --------
        * filename: a string, indicating the path and name of a CSV file, which contains the webpage data.
    ---- Outputs: --------
        * X: a dataframe of all the webpages, where each row represents a webpage.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def load_webpages(filename):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    X = pd.read_csv(filename)
    #########################################
    return X
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_load_webpages
	OR 
        nosetests -v test1.py:test_load_webpages
        ---------------------------------------------------
    '''
    
    

#----------------------------------------------------
'''
    (Count Word Frequency) Given the dataframe X, compute the frequency of the keywords in each webpage's
    'Description' and add a column (called 'Count') to the dataframe X to store the frequencies of the 
    searched keyword. 
	 For example, if the searched keyword is 'thanksgiving', and we have the following webpages: 
	 Webpage 0: Thanksgiving is coming 
	 Webpage 1: This is my thanksgiving recipe for my thanksgiving dinner 
	 Webpage 2: Hello world 
	 Then the result dataframe X should be as follows: 
	    ID  | URL | ...| Count 
	  ----------------------- 
	    0   | ... | ...| 1 
	    1   | ... | ...| 2 
	    2   | ... | ...| 0 
	  ----------------------- 
	 Note, the keyword matching is not case-sensitive. So we assume that all words in the webpages should be 
     converted to lower cases before being matched with the keyword. For example, in webpage 0, the 
     'Thanksgiving' should be first converted to lower cases 'thanksgiving', then it will match with the 
     keyword 'thanksgiving' and being counted. 
    ---- Inputs: --------
        * X: a dataframe of all the webpages, where each row represents a webpage.
        * keyword: the searched keyword, a string, here we assume the string contains only one word that is 
        being searched.
    ---- Outputs: --------
        * X: a dataframe of all the webpages, where each row represents a webpage.
    ---- Hints: --------
        * In a pandas Series x (which is a column of a dataframe X), you could find various functions for
          x.str to process string values, for example if a function is named f(), you could call the function 
          by x.str.f(). 
        * To ignore the cases in word count, you could use a flag value in the "re" package as the parameter 
          for the above f() function. Here "re" is designed for regular expression. 
        * This problem can be solved using 1 line(s) of code.
'''
#I need to add a column 'Count' that is a count of a given key work in a dataframe row. Not true.
#They want the count to be of that keyword just in one column, 'Description'.
#---------------------
def count_word_frequency(X, keyword):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    X['Count'] = list(X.Description.str.count(keyword, re.M|re.I))
    #########################################
    return X
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_count_word_frequency
	OR 
        nosetests -v test1.py:test_count_word_frequency
        ---------------------------------------------------
    '''
    
    ''' 
    If you have passed the above test, the result data should have been saved into a file, named 
    "data_X1.csv". You could take a look at this file for the result.
    
    '''
    

#----------------------------------------------------
'''
    (Rank Webpages by Word Frequency) Given the dataframe (X1), rank all the webpages by descending order of 
    Keyword frequency (Count). 
    ---- Inputs: --------
        * X1: a dataframe of all the webpages containing the column of word frequency, where each row 
          represents a webpage.
    ---- Outputs: --------
        * R1: a dataframe of all the webpages, where the webpages are sorted according to descending order of 
          word frequency.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def rank_word_frequency(X1):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    R1 = X1.sort_values(by='Count', ascending=False)
    #########################################
    return R1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_rank_word_frequency
	OR 
        nosetests -v test1.py:test_rank_word_frequency
        ---------------------------------------------------
    '''
    
    ''' 
    If you have passed the above test, the result data should have been saved into a file, named "data_R1.csv".
    You could take a look at this file for the ranking result.
    (Demo) Now you can check out the search engine that you have built by typing the following in the 
    terminal: python3 demo1.py
    Then open the browser and view the URL showed in the terminal, for example: http://127.0.0.1:5000/
    
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
        ----------- Problem 1 (15 points in total)--------------------- ... ok
        * (5 points) load_webpages ... ok
        * (5 points) count_word_frequency ... ok
        * (5 points) rank_word_frequency ... ok
        ----------------------------------------------------------------------
        Ran 3 tests in 0.006s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* filename:  a string, indicating the path and name of a CSV file, which contains the webpage data. 
* X:  a dataframe of all the webpages, where each row represents a webpage. 
* keyword:  the searched keyword, a string, here we assume the string contains only one word that is being 
  searched. 
* X1:  a dataframe of all the webpages containing the column of word frequency, where each row represents a 
  webpage. 
* R1:  a dataframe of all the webpages, where the webpages are sorted according to descending order of word 
  frequency. 

'''
#--------------------------------------------