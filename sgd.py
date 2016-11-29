import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import tokenize
from scipy.optimize import fmin_bfgs
from scipy import optimize
import math
import operator as op
import time

N=300000
alpha = 50.0
epsilon = 0.00001
batchsize = 256
lam = 0.000001


'''
add regularizer,
add single w
add SGD
'''
time1 = time.time()

with open('copy.json', 'r') as file_req:
	data_extraction = file_req.readlines()[0:N]
		
data_extraction = map(lambda p: p.rstrip(), data_extraction)
data_json_str = "[" + ','.join(data_extraction) + "]"
df = pd.read_json(data_json_str)

x = []
y = [[] for q in range(N)]
for i in range(0,N):
	if df['stars'][i] != 3:
		x.append(df['text'][i])
		if df['stars'][i] > 3:
			y[i].append([1])
		else:
			y[i].append([0])
	else:
		pass

y = [q for q in y if q!=[]]
#count_vector = CountVectorizer()
count_vector = CountVectorizer(encoding=u'utf-8', stop_words='english', max_features=10000)
X = count_vector.fit_transform(x).toarray()



ones = [1] * len(X);
ones = np.asarray(ones);
ones = ones.T;
ones = ones.reshape((X.shape[0], 1))

X = np.concatenate((ones, X), axis = 1)
y = np.array(y)
y = np.reshape(y,(-1,1))
print " total data  ", X.shape

size = y.shape[0]
k = int((size/float(5))*4)
print k, size, X.shape
train_X = X[:k,:]
train_y = y[:k,:]

test_X = X[k:,:]
test_y = y[k:,:]

print " training data  ", train_X.shape
print " testing data ", test_X.shape

print "time taken for preprocessing is ", time.time() - time1

datapoints = train_X.shape[0]
w = np.zeros((train_X.shape[1],1))

def sigmoid_func(w, x):
	w = w.reshape((train_X.shape[1],1))
	act = -np.dot(x, w)
	exp_act = np.exp(act)
#	return .5 * (1 + np.tanh(.5 * act))
	return 1/(1+exp_act)



def gradient(w,x,y):
	sigm = sigmoid_func(w,x);
        temp = sigm - y;
        temp = temp*x
        grad = temp.sum(axis=0).reshape(w.shape)
	grad /= float(datapoints)
	grad = grad + ((lam * w)/len(x))
	return grad

def classify(w, x):
	prob = sigmoid_func(w, x)
        out = []
        for i in range(0, len(prob)):
            out.append(1 if prob[i] >= 0.5 else 0)
        
	return out

def neg_log_likelihood(w, x, y):
	sigm = sigmoid_func(w,x)
        c1_ll = y*np.log(sigm + 1e-50)
        c2_ll = (1-y)*np.log(1-sigm+1e-50)
        log_likelihood = c1_ll + c2_ll
	neg_log = -np.mean(log_likelihood)
	neg_log = neg_log + (lam * np.sum((w*w)/(2*len(x))))
	return neg_log

def stoc_grad_desc(x, y):
    w = np.zeros((train_X.shape[1],1), dtype=np.uint32)
    nll_prev = neg_log_likelihood(w, x, y)
    for i in range(0,25):
        k = 0
        while k < len(x)-batchsize+1:
            w = w - (alpha * gradient(w, x[k:k+batchsize,:], y[k:k+batchsize, :]))
            k = k + batchsize
#        w = w - (alpha * gradient(w, x, y))
        nll = neg_log_likelihood(w, x, y) 
        diff = np.abs(nll - nll_prev)
        nll_prev = nll;
        print nll
        if diff < epsilon :
            print diff
            print "iteration no : ", i
            break

    print "iteration no : ", i
    return w

#ret = fmin_bfgs(neg_log_likelihood, np.zeros((train_X.shape[1],1)), fprime=gradient, args=(train_X,train_y), full_output=True)
#ret = optimize.fmin_l_bfgs_b(neg_log_likelihood, np.zeros((train_X.shape[1],1)), fprime=gradient, args=(train_X,train_y))

#res = ret[0].reshape((train_X.shape[1],1))

res = stoc_grad_desc(train_X, train_y)
out = classify(res, test_X)
print np.transpose(out)


count = 0
for i in range(0, len(test_y)):
	if test_y[i] == 1 and out[i] == 1 :
		count = count +1;
	else:
		if test_y[i] == 0 and out[i] == 0 :
			count = count + 1;

print " accuracy is ", (count/float(len(test_y)))*100

