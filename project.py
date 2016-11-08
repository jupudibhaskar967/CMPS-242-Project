import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import tokenize
from scipy.optimize import fmin_bfgs
import math

N=10

with open('data.json', 'r') as file_req:
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
			y[i].append([1,0])
		else:
			y[i].append([0,1])
	else:
		pass

y = [q for q in y if q!=[]]
count_vector = CountVectorizer()
X = count_vector.fit_transform(x).toarray()
y = np.array(y)
y = np.reshape(y,(-1,2))

train_X = X[0:6,:]
train_y = y[0:6,:]

test_X = X[6:N,:]
test_y = y[6:N,:]

datapoints = train_X.shape[0]
w = np.zeros((train_X.shape[1],2))


probes = 0;
def softmax(w, x):
	w = w.reshape((train_X.shape[1],2))
	act = np.dot(x, w)
	act -= act.max(axis=1)[:, np.newaxis]
	exp_act = np.exp(act)
	fin_res = exp_act/exp_act.sum(axis=1)[:,np.newaxis]
	if np.sum(np.isnan(fin_res)== True) > 1:
		print "softmax becomes numerically unstable", np.sum(np.isnan(fin_res)==True)
		raise
	return fin_res


def gradient(w,x,y):
	sig_activation = softmax(w,x);
	temp = sig_activation * y
	temp = temp/temp.sum(axis=1)[:,np.newaxis]
	sig_activation -= temp
	grad = np.dot(x.T, sig_activation)
	grad /= float(datapoints)
	grad = grad.ravel()
	return grad

def classify(w, x):
	prob = softmax(w, x)
	#prob = np.dot(x, w)
	probes = prob
	prob = np.argmax(prob, axis=1).squeeze()
	return prob

def neg_log_likelihood(w, x, y):
	sig_activation = softmax(w,x)
	temp = sig_activation*y
	temp = temp.sum(axis=1)
	#print temp
	neg_log = -np.mean(np.log(temp))
	return neg_log



np.random.seed()
ret = fmin_bfgs(neg_log_likelihood, np.zeros((train_X.shape[1],2)), fprime=gradient, args=(train_X,train_y), full_output=True)
res = ret[0].reshape((train_X.shape[1],2))
out = classify(res, test_X)
print out

print test_y
