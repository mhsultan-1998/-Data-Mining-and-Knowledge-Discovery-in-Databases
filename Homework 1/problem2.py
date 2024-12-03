import numpy as np
import pandas as pd
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 2: Getting familiar with pandas package (27 points)
     In this problem, we will get familiar with the Pandas python package.  Pandas is the library for tabular data analysis in Python.  It provides fast, flexible, and expressive data structures designed to make
     working with tabular and multidimensional data both easy and intuitive.  You could read the tutorials for Pandas: 
	 https://www.learndatasci.com/tutorials/python-pandas-tutorial-complete-introduction-for-beginners/ 
	 https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html 
    -------------------------
    Package(s) to Install:
        Please install python version 3.7 or above and the following package(s):
        * pandas (for tabular data analysis)
    How to Install:
        * pandas: To install 'pandas' using pip, you could type in the terminal: 
            pip3 install pandas
    -------------------------
    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    Create the following data frame with 2 columns (height and width) and 3 rows/records using Pandas: 
	|'height'| 'width' | 
	|--------|---------| 
	|    1   |    4    | 
	|    2   |    5    | 
	|    3   |    6    |. 
    ---- Outputs: --------
        * X: a pandas dataframe.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def dataframe():
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    data = {'height': [1,2,3], 'width': [4,5,6]}
    X = pd.DataFrame(data)
    #########################################
    return X
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_dataframe
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    Load a data frame from CSV file. The CSV file contains a header line (the first row), 
    indicating the names of all the columns. 
    ---- Inputs: --------
        * filename: the file name of a CSV file, a string.
    ---- Outputs: --------
        * X: a pandas dataframe.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def load_csv(filename='A.csv'):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    X = pd.read_csv(filename)
    #print(X)
    #########################################
    return X
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_load_csv
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    Save a data frame (X) into a CSV file (filename). Note the CSV file should NOT contain the index column. 
    ---- Inputs: --------
        * X: a pandas dataframe.
        * filename: the file name of a CSV file, a string.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def save_csv(X, filename = 'A.csv'):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    # directory = 'C:\Users\mirha\Desktop\BCB4003\Homework 1\'' + filename 
    # print(directory)
    # X.to_csv(r'C:\Users\mirha\Desktop\BCB4003\Homework 1\A.csv', index = False)
    X.to_csv(filename, index=False)
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_save_csv
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    filter all the data records in a dataframe with height (column) greater or equals to a threshold value. 
    For example, suppose we have the following dataframe as the input: 
	|'height'| 'width' | 
	|--------|---------| 
	|    0   |    4    | 
	|    1   |    5    | 
	|    2   |    6    | 
	|    3   |    7    | 
    If the threshold is 2, then all the rows with height greater or equal to 2 will be returned. 
    The result will be another data frame: 
	|'height'| 'width' | 
	|--------|---------| 
	|    2   |    6    | 
	|    3   |    7    |. 
    ---- Inputs: --------
        * X: a pandas dataframe.
        * t: an integer scalar, the threshold of the height.
    ---- Outputs: --------
        * Xt: the result dataframe, containing only the records with heights greater or equals to the threshold t.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def filter_height(X, t):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    Xt = X.copy()
    rowN = len(Xt.index)
    g = 0
    while g < rowN:
        if Xt['height'][g] < t:
            Xt = Xt.drop(g)
        g += 1
    #########################################
    return Xt
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_filter_height
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    Group the data records in a dataframe according to a column (k). Then sum all the values within each group. Suppose we have the following data frame X: 
	| 'ID'   | 'count' | 
	|--------|---------| 
	|    1   |    4    | 
	|    1   |    5    | 
	|    2   |    6    | 
	|    2   |    7    | 
 We have duplicated values in ID column. Now we want to sum the 'count' values within each group of data 
 records with the same ID's. So that the records with ID=1, should have a sum of 'count' = 4+5 = 9 and the
 records with ID=2, should have a sum of 'count' = 6+7 = 13 The output should be: 
	| 'ID'   | 'count' | 
	|--------|---------| 
	|    1   |    9    | 
	|    2   |    13   |. 
    ---- Inputs: --------
        * X: a pandas dataframe.
        * k: a string indicating the name of a column in a data frame.
    ---- Outputs: --------
        * Y: a pandas dataframe.
    ---- Hints: --------
        * you could use the groupby() function of pandas and solve this problem using two line of code. 
        * To convert an index into a column, you could use reset_index() method in pandas. 
        * This problem can be solved using 2 line(s) of code.
'''
#---------------------
def group_sum(X, k):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    hg = X.columns.tolist()    
    gh1 = X.columns.tolist()
    gh1.remove(k)
    gk = X.groupby(k)
    data1 = {}
    for x in hg:
        if x == k:
            data1[x] = list(gk.groups.keys())
        if x != k:
            data1[x] = []       
    for x in gk.groups.keys():
        fg = gk.get_group(x)
        for y in gh1:
            d = data1[y] 
            d.append(int(sum(fg[y].tolist()))) 
            #sum function here, can be different elsewhere hence why this
            #is longer than it is expected to be.
            data1[y] = d  
    Y = X.drop(X.index[len(gk.groups.keys()):])
    df6 = pd.DataFrame(data1)
    Y.update(df6)
    #########################################
    return Y
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_group_sum
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    Merge two data frames on a column (k) into one joint data frame. Suppose we have the following data frame X: 
	| 'ID'   | 'count' | 
	|--------|---------| 
	|    1   |    9    | 
	|    2   |    13   |

  and we have another data frame  Y: 
	| 'ID'   | 'name'  | 
	|--------|---------| 
	|    1   | 'Alex'  | 
	|    2   | 'Bob'   | 
	|    3   | 'Tom'   | 
 Merge the two tables with k='ID'. The output should be: 
	| 'ID'   | 'count' |  'name'  | 
	|--------|---------| ---------| 
	|    1   |    9    |   'Alex' | 
	|    2   |    13   |   'Bob'  | 
 In database, there are different ways of joining two tables: 'inner join', 'left join', 'right join', 'outer join'. Here we using 'inner join' of the two tables to get the result table (dataframe). 
 In cases when the two tables have columns with identical names, suffixes '_x', '_y' should be added in the column names of the result table. For examples, suppose we have a data frame X: 
	| 'ID'   | 'name'  | 
	|--------|---------| 
	|    1   | 'Smith' | 
	|    2   | 'Wilson'|

  and we have another data frame  Y: 
	| 'ID'   | 'name'  | 
	|--------|---------| 
	|    1   | 'Alex'  | 
	|    2   | 'Bob'   | 
	|    3   | 'Tom'   | 
 Now in both data frames, we have a column called 'name'. After merging the two tables with k='ID'. The output should be: 
	| 'ID'   | 'name_x'|  'name_y'| 
	|--------|---------| ---------| 
	|    1   | 'Smith' |   'Alex' | 
	|    2   | 'Wilson'|   'Bob'  | . 
    ---- Inputs: --------
        * X: a pandas dataframe.
        * Y: a pandas dataframe.
        * k: a string indicating the name of a column in a data frame.
    ---- Outputs: --------
        * J: the result pandas dataframe after joining two dataframes.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def merge(X, Y, k):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    Xc = list(X.columns)
    Yc = list(Y.columns)
    for x in Xc:
        d = Xc.index(x)
        if x in Yc and x != k:
            c = x
            X.columns.values[d] = c + "_x"
            e = Yc.index(x)
            Y.columns.values[e] = c + "_y"
    J = pd.merge(X, Y, on=k)
    #########################################
    return J
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_merge
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    Given a dataframe X and a column (k), sort data records in descending order of the values in the column (k) . Suppose we have the following data frame X: 
	|'height'| 'width' | 
	|--------|---------| 
	|    1   |    6    | 
	|    2   |    4    | 
	|    3   |    5    | 
 If we want to sort the table with values in column k='height' in descending order, the output should be: 
	|'height'| 'width' | 
	|--------|---------| 
	|    3   |    5    | 
	|    2   |    4    | 
	|    1   |    6    | 
 If we want to sort the table with values in column k='width' in descending order, the output should be: 
	|'height'| 'width' | 
	|--------|---------| 
	|    1   |    6    | 
	|    3   |    5    | 
	|    2   |    4    |. 
    ---- Inputs: --------
        * X: a pandas dataframe.
        * k: a string indicating the name of a column in a data frame.
    ---- Outputs: --------
        * Y: a pandas dataframe.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------

def sort_values(X, k):
    #########################################
    Y = X.sort_values(by=k, ascending=False)
    #########################################
    return Y
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_sort_values
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    Given a dataframe X and two columns (k, l), create a pandas series, where each element is the value in
    column k divided by column l. Suppose we have the following data frame X: 
	|'height'| 'width' | 
	|--------|---------| 
	|    1   |    6    | 
	|    2   |    4    | 
	|    3   |    5    | 
 If we want to divide column k='width' by column l='height', the output should be a pandas series: 
	|   5/1   | 
	|   4/2   | 
	|   6/3   | 
 If we want to divide column k='height' by column l='width', the output should be a pandas series: 
	|  1/5   | 
	|  2/4   | 
	|  3/6   |. 
    ---- Inputs: --------
        * X: a pandas dataframe.
        * k: a string indicating the name of a column in a data frame.
        * l: a string indicating the name of a column in a data frame.
    ---- Outputs: --------
        * Y: a pandas dataframe.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def divide(X, k, l):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    L1 = []
    kc = list(X[k])
    lc = list(X[l])
    z = 0
    while z < len(kc):
        L1.append(kc[z]/lc[z])
        z += 1
    Y = pd.Series(L1)
    #########################################
    return Y
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_divide
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    Given a dataframe X, a pandas series y and a column name (k), create a new column in X as name k, insert 
    the series y into the new column. Suppose we have the following data frame X: 
	|'height'| 'width' | 
	|--------|---------| 
	|    1   |    6    | 
	|    2   |    4    | 
	|    3   |    5    | 
 and a pandas series y: 
	|  10 | 
	|  20 | 
	|  30 | 
 If we want to insert the series y into X with a column name 'depth', the output should be a pandas series: 
	|'height'| 'width' | 'depth' | 
	|--------|---------|---------| 
	|    1   |    6    |   10    | 
	|    2   |    4    |   20    | 
	|    3   |    5    |   30    |. 
    ---- Inputs: --------
        * X: a pandas dataframe.
        * y: a pandas series, which is a column of a dataframe.
        * k: a string indicating the name of a column in a data frame.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def insert_column(X, y, k):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    X[k] = y
    return X
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py:test_insert_column
        ---------------------------------------------------
    '''
    

#--------------------------------------------

''' 
    TEST problem 2: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test2.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 2 (27 points in total)--------------------- ... ok
        * (3 points) dataframe ... ok
        * (3 points) load_csv ... ok
        * (3 points) save_csv ... ok
        * (3 points) filter_height ... ok
        * (3 points) group_sum ... ok
        * (3 points) merge ... ok
        * (3 points) sort_values ... ok
        * (3 points) divide ... ok
        * (3 points) insert_column ... ok
        ----------------------------------------------------------------------
        Ran 9 tests in 0.006s

        OK
'''

#--------------------------------------------


#--------------------------------------------
'''
    List of All Variables 

* X:  a pandas dataframe. 
* Y:  a pandas dataframe. 
* y:  a pandas series, which is a column of a dataframe. 
* J:  the result pandas dataframe after joining two dataframes. 
* filename:  the file name of a CSV file, a string. 
* k:  a string indicating the name of a column in a data frame. 
* l:  a string indicating the name of a column in a data frame. 
* s:  an integer scalar, the sum of the values in the column of the data frame. 
* v:  a list of values. 
* t:  an integer scalar, the threshold of the height. 
* Xt:  the result dataframe, containing only the records with heights greater or equals to the threshold t. 

'''
#--------------------------------------------