# CMPS-242-Project
- Contributions.txt file contains the contributions made by each member for this project.


- Sentiment Analysis of yelp reviews using Logistic regression :

The folder Binary_Classification contains the code for Sentiment Analysis:

1. logistic_regression_fmin.py - Logistic regression using fmin_l_bfgs_b as optimization technique
2. logistic_regression_sgd.py  - Logistic regression using SGD as optimization technique
3. naive_bayes.py	       - Naive Bayes approach for classification


- Predict ratings based on reviews :

The folder Multi_Classification contains the code to predict ratings

1. multi_cl_log_reg_fmin.py - Multiclass Logistic regression using fmin_l_bfgs_b as optimization technique
2. multi_cl_log_reg_sgd.py  - Multiclass Logistic regression using SGD as optimization technique
3. multi_naive_bayes.py     - Multiclass Naive Bayes approach for classification

HOWTO:

You need to have yelp_academic_dataset_review.json file in the CMPS-242-Project folder to run these files.
You need to modify the value of N inside these python files to specify the size of data (number of reviews)
If you are using logistic_regression_sgd.py / multi_cl_log_reg_sgd.py , you need to tune the parameter alpha for better performance.
To run any python file in the code base, type 'python file_name.py'

UNIT TEST CASES:
Unit test cases are present in unit-tests folder.
To run any python file in this folder, type 'python file_name.py'
The test file names and their utility are same as the files described above with an 'test' appended to their names
