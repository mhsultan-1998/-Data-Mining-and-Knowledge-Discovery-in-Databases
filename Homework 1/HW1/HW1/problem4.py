import numpy as np
import pandas as pd
from problem3 import *
from problem2 import *
from problem1 import *
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 4: (Moneyball) Player Ranking for Oakland A's Baseball Team (OAK) in year 2002 (27 points)
    In this problem, you will rank baseball players for Oakland A's using different methods
    A list of all variables being used in this problem is provided at the end of this file.
'''

#----------------------------------------------------
'''
    (Compute Players' Batting Average) Given a dataframe (X6) of all the candidate players, compute the 
     Batting Average (BA) of each player and insert a new column named 'BA' to store the data. 
    ---- Inputs: --------
        * X6: a dataframe containing all the candidate players for evaluation in the year (2001), who have 
        sufficient experience (at least min_AB) and affordable price tag (at most max_salary).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_players_BA(X6):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    X6['BA'] =  divide(X6, 'H', 'AB')
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_players_BA
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    If you have passed the previous test case, the result data frame should have been saved into a file, 
    called 'moneyball_X7.csv'.  Now let's rank all of these players using the traditional metric: Batting 
    Average. (Rank Players by BA) Given a dataframe (X7) of all the candidate players, rank all the players 
    based upon descending order of Batting Average (BA). If you have solved this problem and passed the test 
    case, the result data frame will be saved into a file, called 'moneyball_R1.csv'. You could take a look at
    the ranking results of the players. 
    ---- Inputs: --------
        * X7: a dataframe containing all the candidate players info, including Batting Average (BA).
    ---- Outputs: --------
        * R1: a dataframe containing candidate players ranked by descending order of Batting Average (BA).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def rank_players_BA(X7):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    R1 = sort_values(X7, 'BA')
    #########################################
    return R1
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_rank_players_BA
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    (Compute Players' On-Base Percentage) Given a dataframe (X7) of all the candidate players, compute the
     (OBP) of each player and insert a new column named 'OBP' to store the data. 
    ---- Inputs: --------
        * X7: a dataframe containing all the candidate players info, including Batting Average (BA).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_players_OBP(X7):
    #########################################
    ## INSERT YOUR CODE HERE (4 points) 
    L1 = []
    H = list(X7['H'])
    AB = list(X7['AB'])
    BB = list(X7['BB'])
    HBP = list(X7['HBP'])
    SF = list(X7['SF'])
    z = 0
    while z < len(H):
        L1.append(compute_OBP(H[z], AB[z], BB[z], HBP[z], SF[z]))
        z += 1
    X7['OBP'] = pd.Series(L1)
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_players_OBP
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    If you have passed the previous test case, the result data frame should have been saved into a file, 
    called 'moneyball_X8.csv'.  Now let's rank all of these players using the new metric: On-Base Percentage. 
    (Rank Players by OBP) Given a dataframe (X8) of all the candidate players, rank all the players based upon
    descending order of On-Base Percentage (OBP). 
    If you have solved this problem and passed the test case, the result data frame will be saved into a file, 
    called 'moneyball_R2.csv'. You could take a look at the ranking results of the players. 
    ---- Inputs: --------
        * X8: a dataframe containing all the candidate players info, including Batting Average (BA) and On-Base Percentage (OBP).
    ---- Outputs: --------
        * R2: a dataframe containing candidate players ranked by descending order of On-Base Percentage (OBP).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def rank_players_OBP(X8):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    R2 = sort_values(X8, 'OBP')
    #########################################
    return R2
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_rank_players_OBP
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    (Compute Players' 1B) Given a dataframe (X8) of all the candidate players, compute the number of singles 
     (1B) of each player and insert a new column named '1B' to store the data. 
    ---- Inputs: --------
        * X8: a dataframe containing all the candidate players info, including Batting Average (BA) and 
        On-Base Percentage (OBP).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_players_1B(X8):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    L1 = []
    H = list(X8['H'])
    BB = list(X8['2B'])
    BBB = list(X8['3B'])
    HR = list(X8['HR'])
    z = 0
    while z < len(H):
        L1.append(H[z] - (BB[z] + BBB[z] + HR[z]))
        z += 1
    X8['1B'] = pd.Series(L1)
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_players_1B
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    If you have passed the previous test case, the result data frame should have been saved into a file, 
    called 'moneyball_X9.csv'.  Now let's compute the total number bases for each player. 
    (Compute Players' TB) Given a dataframe (X9) of all the candidate players, compute the total number of
     bases (TB) of each player and insert a new column named 'TB' to store the data. 
    ---- Inputs: --------
        * X9: a dataframe containing all the candidate players info, including Batting Average (BA), On-Base
        Percentage (OBP) and Singles (1B).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_players_TB(X9):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    L1 = []
    B = list(X9['1B'])
    BB = list(X9['2B'])
    BBB = list(X9['3B'])
    HR = list(X9['HR'])
    z = 0
    while z < len(B):
        L1.append(compute_TB(B[z], BB[z], BBB[z], HR[z]))
        z += 1
    X9['TB'] = pd.Series(L1)
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_players_TB
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    If you have passed the previous test case, the result data frame should have been saved into a file,
    called 'moneyball_X10.csv'.  Now let's compute the total number bases for each player. 
    (Compute Players' SLG) Given a dataframe (X10) of all the candidate players, compute the slugging 
    percentage (SLG) of each player and insert a new column named 'SLG' to store the data. 
    ---- Inputs: --------
        * X10: a dataframe containing all the candidate players info, including Batting Average (BA), 
        On-Base Percentage (OBP) Singles (1B), and Total Bases (TB).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def compute_players_SLG(X10):
    #########################################
    ## INSERT YOUR CODE HERE (4 points)
    L1 = []
    TB = list(X10['TB'])
    AB = list(X10['AB'])
    z = 0
    while z < len(AB):
        L1.append(compute_SLG(TB[z], AB[z]))
        z += 1
    X10['SLG'] = pd.Series(L1) 
    #########################################
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_compute_players_SLG
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    If you have passed the previous test case, the result data frame should have been saved into a file, 
    called 'moneyball_X11.csv'.  Now let's rank all of these players using the new metric: Slugging Percentage.
    (Rank Players by SLG) Given a dataframe (X11) of all the candidate players, rank all the players based upon
    descending order of Slugging Percentage (SLG). 
    If you have solved this problem and passed the test case, the result data frame will be saved into a file,
    called 'moneyball_R3.csv'. You could take a look at the ranking results of the players. 
    ---- Inputs: --------
        * X11: a dataframe containing all the candidate players info, including Batting Average (BA), 
        On-Base Percentage (OBP) Singles (1B), Total Bases (TB) and Slugging Percentage (SLG).
    ---- Outputs: --------
        * R3: a dataframe containing candidate players ranked by descending order of Slugging Percentage (SLG).
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def rank_players_SLG(X11):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    R3 = sort_values(X11, 'SLG')
    #########################################
    return R3
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py:test_rank_players_SLG
        ---------------------------------------------------
    '''
    

#--------------------------------------------

''' 
    TEST problem 4: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test4.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 4 (27 points in total)--------------------- ... ok
        * (4 points) compute_players_BA ... ok
        * (3 points) rank_players_BA ... ok
        * (4 points) compute_players_OBP ... ok
        * (3 points) rank_players_OBP ... ok
        * (3 points) compute_players_1B ... ok
        * (3 points) compute_players_TB ... ok
        * (4 points) compute_players_SLG ... ok
        * (3 points) rank_players_SLG ... ok
        ----------------------------------------------------------------------
        Ran 8 tests in 0.006s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* X6:  a dataframe containing all the candidate players for evaluation in the year (2001), who have sufficient experience (at least min_AB) and affordable price tag (at most max_salary). 
* X7:  a dataframe containing all the candidate players info, including Batting Average (BA). 
* X8:  a dataframe containing all the candidate players info, including Batting Average (BA) and On-Base Percentage (OBP). 
* X9:  a dataframe containing all the candidate players info, including Batting Average (BA), On-Base Percentage (OBP) and Singles (1B). 
* X10:  a dataframe containing all the candidate players info, including Batting Average (BA), On-Base Percentage (OBP) Singles (1B), and Total Bases (TB). 
* X11:  a dataframe containing all the candidate players info, including Batting Average (BA), On-Base Percentage (OBP) Singles (1B), Total Bases (TB) and Slugging Percentage (SLG). 
* R1:  a dataframe containing candidate players ranked by descending order of Batting Average (BA). 
* R2:  a dataframe containing candidate players ranked by descending order of On-Base Percentage (OBP). 
* R3:  a dataframe containing candidate players ranked by descending order of Slugging Percentage (SLG). 

'''
#--------------------------------------------