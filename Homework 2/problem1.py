import random
from flask import Flask
from jinja2 import Template,Environment,FileSystemLoader
app= Flask(__name__) # create an app for the website
# Note: please don't import any new package. You should solve this problem using only the package(s) above.
#-------------------------------------------------------------------------

'''
    Problem 1: Getting familiar with flask package (25 points)
     In this problem, we will get familiar with the flask python package.  
     Flask is a library for website development in Python.  It provides fast, flexible, and easy website 
     design framework.  We will use flask to build a simple website. You could read the tutorials for Flask:
	 https://flask.palletsprojects.com/en/1.1.x/quickstart/
    A list of all variables being used in this problem is provided at the end of this file.
'''

#--------------------------
def Terms_and_Conditions():
    ''' 
        By submitting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due. For example, 
       sending your code segment to another student, putting your solution online or lending your laptop (if 
       your laptop contains your solution or your Dropbox automatically copied your solution from your desktop 
       computer and your laptop) to another student to work on this homework will violate this term.
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
    (Hello World Page) In the first URL ('/'), let's create a simple page: when a user visited the homepage 
    of our website, we just return a string 'Hello World!'. This is a static homepage, where the content of 
    the webpage is always the same. 
    ---- Outputs: --------
        * webpage: a webpage returned to the user's browser when a user visits the URL, a string of plain text 
        or html.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
@app.route("/")
def hello_page():
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    webpage = 'Hello World!'
    #########################################
    return webpage
    #-----------------    
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_hello_page
        ---------------------------------------------------
    '''
    
# -------------------------------------------------------
# (Demo) Now you can test the website that you have built by typing the following in the terminal: 
    #python3 demo1.py
# -------------------------------------------------------
#----------------------------------------------------
'''
    (Random Number Page) In the second URL ('/rand'), let's create a simple dynamic page and send each user 
    some different data in the webpage: Each time when a user visits this page, randomly generate a number 
    (between 0 and 1) and return the string of the number back to the user. So when users visit this page 
    multiple times, the user will see a different random number each time. 
    ---- Outputs: --------
        * webpage: a webpage returned to the user's browser when a user visits the URL, a string of plain text 
        or html.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
@app.route("/rand")
def rand_page():
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    webpage = str(random.uniform(0, 1))
    #########################################
    return webpage
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        nosetests -v test1.py:test_rand_page
        ---------------------------------------------------
    '''
    
# -------------------------------------------------------
# (Demo) Now you can test the website that you have built by typing the following in the terminal: python3 demo1.py
# Open the browser to visit the website with relative path "/rand"
# -------------------------------------------------------
#----------------------------------------------------
'''
    (Vote Page) In the third URL ('/vote/<ID>'), let's create a simple webpage for users to send data to the 
    website host: Each time when a user visits this page with a number (ID) in the URL (for example, /vote/10),
    return a webpage with 'Thank you for voting <ID>' (for example, 'Thank you for voting 10'). 
    ---- Inputs: --------
        * ID: the ID of a user votes for, an integer scalar.
    ---- Outputs: --------
        * webpage: a webpage returned to the user's browser when a user visits the URL, a string of plain text
        or html.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
@app.route("/vote/<int:ID>")
def vote(ID):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    webpage = 'Thank you for voting ' + str(ID)
    #########################################
    return webpage
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_vote
        ---------------------------------------------------
    '''
    
# -------------------------------------------------------
# (Demo) Now you can test the website that you have built by typing the following in the terminal: python3 demo1.py
# Open the browser to visit the website with relative path "/vote/2"
# -------------------------------------------------------
#----------------------------------------------------
'''
    (Template) create a jinja2 template that can be used to render 'Hello, <username>!' In this template, we 
    need to have a variable named 'username'. For example, if the username ='Alex', when rendering this 
    template, the returned string will be 'Hello, Alex!'; If the username = 'Bob', when rendering this 
    template, the returned string will be 'Hello, Bob!'. 
    ---- Outputs: --------
        * t: a jinja2 template, which can be used to render a dynamic webpage.
    ---- Hints: --------
        * This problem can be solved using 1 line(s) of code.
'''
#---------------------
def create_template():
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    t = Template("Hello, {{ username }}!")
    #########################################
    return t
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_create_template
        ---------------------------------------------------
    '''
    
#----------------------------------------------------
'''
    load a jinja2 template from a file. 
    ---- Inputs: --------
        * filename: the filename of the template to be used for rendering webpage response, a string.
    ---- Outputs: --------
        * t: a jinja2 template, which can be used to render a dynamic webpage.
    ---- Hints: --------
        * You could user FileSystemLoader in jinja2 to create an Environment in the current folder (".") and 
        use the created environment to load the template from the file. 
        * This problem can be solved using 3 line(s) of code.
'''
#---------------------
def load_template(filename):
    #########################################
    ## INSERT YOUR CODE HERE (5 points)
    # file_loader = FileSystemLoader(filename)
    # env = Environment(loader=file_loader)
    # t = env.get_template()
    templateLoader = FileSystemLoader(searchpath=".")
    templateEnv = Environment(loader=templateLoader)
    t = templateEnv.get_template(filename)
    #########################################
    return t
    #-----------------
    '''  
        TEST: Now you can test the correctness of your code above by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py:test_load_template
        ---------------------------------------------------
    '''
    

#--------------------------------------------
#if __name__=="__main__":
#    app.run()

''' 
    TEST problem 1: 
        Now you can test the correctness of all the above functions by typing the following in the terminal:
        ---------------------------------------------------
        python3 -m nose -v test1.py
        ---------------------------------------------------

        If your code passed all the tests, you will see the following message in the terminal:
        ----------- Problem 1 (25 points in total)--------------------- ... ok
        * (5 points) hello_page ... ok
        * (5 points) rand_page ... ok
        * (5 points) vote ... ok
        * (5 points) create_template ... ok
        * (5 points) load_template ... ok
        ----------------------------------------------------------------------
        Ran 5 tests in 0.006s

        OK
'''

#--------------------------------------------





#--------------------------------------------
'''
    List of All Variables 

* webpage:  a webpage returned to the user's browser when a user visits the URL, a string of plain text or html. 
* ID:  the ID of a user votes for, an integer scalar. 
* filename:  the filename of the template to be used for rendering webpage response, a string. 
* t:  a jinja2 template, which can be used to render a dynamic webpage. 

'''
#--------------------------------------------