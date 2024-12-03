# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:12:41 2020

@author: mirha
"""
import numpy as np
import pandas as pd
import sys
import re
import torch as th



def dataframe():
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    data = {'height': [2,8,7], 'width': [4,5,6]}
    X = pd.DataFrame(data)
    #########################################
    #print(X)
    return X

def load_csv(filename='A.csv'):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    X = pd.read_csv(filename)
    #print(X)
    #########################################
    return X

def save_csv(X, filename = 'A.csv'):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    #X.to_csv(r'C:\Users\mirha\Desktop\BCB4003\Homework 1\A.csv', index = False)
    X.to_csv(filename, index=False)
    
def filter_height(X, t):
    Xt = X.copy()
    rowN = len(Xt.index)
    g = 0
    while g < rowN:
        if Xt['height'][g] < t:
            Xt = Xt.drop(g)
        g += 1

    print(Xt)
    return Xt




def merge(X, Y, k):
    Xc = list(X.columns)
    print("Xc:", Xc)
    Yc = list(Y.columns)
    print("Yc:", Yc)
    for x in Xc:
        d = Xc.index(x)
        if x in Yc and x != k:
            c = x
            X.columns.values[d] = c + "_x"
            e = Yc.index(x)
            Y.columns.values[e] = c + "_y"
    J = pd.merge(X, Y, on=k)
    #For fixing identical column names, it'll have to be before the merge function
    return J

def why_type(x):
    return x.item()

def sort_values(X, k):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    Y = X.sort_values(by=k, ascending=False)
    
    #########################################
    return Y

def insert_column(X, y, k):
    #########################################
    ## INSERT YOUR CODE HERE (3 points)
    X[k] = y
    return X

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

#pd.DataFrame({'ID': [1,1,2,2], 'count': [4,5,6,7]})

# Th = pd.DataFrame({'ID': [1,1,2,2], 'count': [4,5,6,7]})


# print(divide(Th, 'ID', 'count'))
    
# data1 = {'height': [0,1,2,3], 'width': [4,5,6,7]}
# X1 = pd.DataFrame(data1)


# X33 = pd.DataFrame({"ID":[1,2], "name":["Smith", "Wilson"]})
# X34 = pd.DataFrame({"ID":[1,2,3], "name":["Alex", "Bob", "Tom"]})

# X35 = pd.DataFrame({"ID":[1,2], 'count':[9,13]})
# X36 = pd.DataFrame({"ID":[1,2,3], "name":["Alex", "Bob", "Tom"]})

# X40 = pd.DataFrame({'height': [1,2,3], 'width': [6,4,5]})
# S60 = pd.Series([10,20,30])

# print(list(X40['height']))


def compute_EA(RA, RB):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    EA = 1/(1 + 10**((RB-RA)/400))
    #########################################
    return EA

G = pd.DataFrame({"win_ID":[0],"lose_ID":[1]})
#print(G)





rd = pd.DataFrame({'ID': [0,1,2,3,4,5,6,7], 'URL': ['0.html','1.html','2.html','3.html','4.html','5.html','6.html','7.html'],
                   'Title': ["Alex\'s Cooking List",'The Bob Chef','The Carol Chef','Derek\'s Mashed Potato Recipe',
                             'Emmy\'s World Famous Thanksgiving Recipe','Frank\'s Thanksgiving Recipe',
                             'Gabriel\'s Thanksgiving Recipe','Helen\'s Thanksgiving Recipe'], 'Description' :
                       ['Here are my favorite Thanksgiving recipes for more expert opinions, check out the Derek chef and Frank Chef',
                        'I learned everything i know from the Derek chef. For the complete list of thanksgiving recipes, check here.',
                        'This is the best thanksgiving recipe!', 'roast the garlic peel the potatoes boil them it\'s time to mash fold everything together',
                        'Kidnap the Derek Chef. Force him to make Mashed Potato for you.', 
                        'Go to the store and buy a container of Mashed Potato. Open it. Thanksgiving Thanksgiving Thanksgiving Thanksgiving Thanksgiving Thanksgiving Thanksgiving Thanksgiving Thanksgiving Thanksgiving Thanksgiving Thanksgiving Thanksgiving Thanksgiving Thanksgiving Thanksgiving Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe Recipe',
                        'I sell canned Mashed Potatoes. Buy from me Perfect for thanksgiving buy for your thanksgiving thanksgiving is coming Frank is my customer',
                        'I learnt this recipe from Derek']})
                        
list(rd.Description.str.count('thanksgiving', re.M|re.I))

rd1 = rd.assign(Count = list(rd.Description.str.count('thanksgiving', re.M|re.I)))






Y = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])


array = Y.reshape(1, 6)


def remove_sink_nodes(A):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    #if all values in a column in the adjacency matrix sum to 0, fill that column with 1s.
    s, r, c, k, ze = list(np.sum(A, axis=0)), list(A.shape)[0], list(A.shape)[1], 0, [1]*list(A.shape)[0]
    while k < len(s):
        if s[k] == 0:
            de = list(A[:, k])
            z = 0
            while z < len(de):
                if de[z] == 0:
                    de[z] = 1
                    z += 1
            A[:, k] = ze
        k += 1
    #########################################
    return A


A1 = np.array( [ [0, 1, 0],
                    [1, 0, 0],
                    [0, 1, 0]])


a2 =  remove_sink_nodes(A1)

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

#print('wtf?!',compute_S(a2))


xew =  np.ones(3)/3




def random_walk_one_step(G, x):
    #########################################
    ## INSERT YOUR CODE HERE (6 points)
    y = np.dot(G, x)
    #########################################
    return y

G7 = np.array([[ 0. ,  0.5,  1. ],
                  [ 0.5,  0. ,  0. ],
                  [ 0.5,  0.5,  0. ]]) # transition matrix of shape (3 by 3)
x32 =  np.ones(3)/3




x17 = np.array([0.1,0.5,1.8,100.9])
y21 = np.array([0.2,0.4,1.85,100.94])
#np.sum(A, axis=1)
x73 = np.array([1,2,np.nan,3,np.nan])

def least_square(X, y):
    #########################################
    ## INSERT YOUR CODE HERE (10 points)
    w = (np.linalg.inv((X.T@X)))@X.T@y
    #########################################
    return w
def ridge_regression(X, y, a=0.0001):
    #########################################
    ## INSERT YOUR CODE HERE (10 points)
    w = (np.linalg.inv((np.add((X.T@X), np.multiply(a,np.identity(X.shape[1]))))))@X.T@y
    #########################################
    return w

def extract_user_j(R_j, I):
    #########################################
    ## INSERT YOUR CODE HERE (7 points)
    X = I[(np.where((np.isnan(R_j)) == False, True, False))]
    y = R_j[(np.where((np.isnan(R_j)) == False, True, False))]
    #########################################
    return X, y


Rj1 = np.array([ 1, np.nan, 2, np.nan])
I7 = np.array([[0.1, 0.2],
                  [0.3, 0.4],
                  [0.5, 0.6],
                  [0.7, 0.8]])





def compute_L(z, y):
    if z >= 1000:
        L = z*(1-y)
    else:
        L = np.log(1 + np.exp(z)) - (y*z)
    return L

import torch.nn.functional as F
# -----------------------------------------------------------
# The class definition of Logistic Regression
class LogisticRegression(th.nn.Module):
    # create an object for logistic regression
    def __init__(self, p):
        super(LogisticRegression, self).__init__()
        # create a linear layer from p dimensional input to 1 dimensional output
        self.layer = th.nn.Linear(p, 1) # the linear layer includes parameters of weights (w) and bias (b)
    # the forward function, which is used to compute linear logits on a mini-batch of samples
    def forward(self, x):
        z = self.layer(x) # use the linear layer to compute z
        return z
    
m = LogisticRegression(3)# create a logistic regression object
m.layer.weight.data = th.tensor([[ 0.5, 0.1,-0.2]])
m.layer.bias.data = th.tensor([0.2]) 
    # create a toy loss function: the sum of all elements in w and b
L = m.layer.weight.sum() + m.layer.bias.sum()

optimizer = th.optim.SGD(m.parameters(), lr=0.1)
optimizer.zero_grad()


#print(optimizer.param_groups[0]['lr'])

wq, bq = optimizer.param_groups[0]['params'][0], optimizer.param_groups[0]['params'][1]
lrq = optimizer.param_groups[0]['lr']
dW, dB = wq.grad_fn, bq.grad_fn
optimizer.zero_grad()

# ds = optimizer.param_groups[0]

# for x in ds:
#     print(x)

def choose_action_exploit(Rt, Ct):
    #########################################
    ## INSERT YOUR CODE HERE (25 points)
    At = np.nan_to_num(np.divide(Rt, Ct))
    a = np.where(At == np.max(At))[0]
    #########################################
    return a

Rt1 = np.array([0.,2.,3.]) # check with 3 possible actions
Ct1 = np.array([0,1,5])

#print('a', choose_action_exploit(Rt1, Ct1))


dictv = {'truth': 185.179993, 'day1': 197.22307753038834, 'day2': 197.26118010160317, 
         'day3': 197.19846975345905, 'day4': 197.1490578795196, 'day5': 197.37179265011116}

output = pd.DataFrame()
#output = output.append(dictv, ignore_index=True)
#print(output)

dictv1 = {'height': 2, 'width': 4}

dictV = {**dictv, **dictv1}
output = output.append(dictV, ignore_index=True)

output['gert'] = [2]
    
#print(output.head())

df3 = pd.DataFrame()
#print("well?", pd.concat([df3, output]))
#print(output['height'][0])




sd = np.array([[ 1 , 0 ,-1 ],
      [-1 , 1 , 0 ],
      [ 1 , 0 ,-1 ]])
    

print(list(np.where(sd == 0)))
gh = np.where(sd == 0)
kj = list(zip(*gh))
print('kj', kj)

print('well?',np.flip(sd, 1))