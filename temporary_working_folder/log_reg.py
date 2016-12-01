import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import tokenize
from scipy.optimize import fmin_bfgs
from scipy import optimize
import math

N=100000
'''
add regularizer,
add single w
add SGD
'''
with open('copy.json', 'r') as file_req:
	data_extraction = file_req.readlines()[0:N]
		
data_extraction = map(lambda p: p.rstrip(), data_extraction)
data_json_str = "[" + ','.join(data_extraction) + "]"
df = pd.read_json(data_json_str)

lam = 0.000001

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

datapoints = train_X.shape[0]
w = np.zeros((train_X.shape[1],1))


probes = 0;

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
#	print (((lam * w)/len(x))), " hi "
#	print (((lam * w)/len(x))), " hello "
       # grad = grad.reshape(w.shape)
	return grad

def classify(w, x):
	prob = sigmoid_func(w, x)
	#prob = np.dot(x, w)
#	prob = np.argmax(prob, axis=1).squeeze()
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
	neg_log = neg_log + (lam * (w*w)/(2*len(x)))
#	print (lam * (w*w)/(2*len(x)))
	return neg_log


#ret = fmin_bfgs(neg_log_likelihood, np.zeros((train_X.shape[1],1)), fprime=gradient, args=(train_X,train_y), full_output=True)
ret = optimize.fmin_l_bfgs_b(neg_log_likelihood, np.zeros((train_X.shape[1],1)), fprime=gradient, args=(train_X,train_y))


res = ret[0].reshape((train_X.shape[1],1))
out = classify(res, test_X)
print np.transpose(out)

#print np.transpose(test_y)
count = 0
for i in range(0, len(test_y)):
	if test_y[i] == 1 and out[i] == 1 :
		count = count +1;
	else:
		if test_y[i] == 0 and out[i] == 0 :
			count = count + 1;

print " accuracy is ", (count/float(len(test_y)))*100

