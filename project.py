import pandas as pd
import numpy as np
from scipy.optimize import fmin_l_bfgs_b	
import math
from multiprocessing.dummy import Pool as ThreadPool
from threading import Thread, current_thread
import multiprocessing

N=100000
bag_of_words=[]
numclasses = 2
with open('yelp_academic_dataset_review.json', 'r') as file_req:
	data_extraction = file_req.readlines()[0:N]
		
data_extraction = map(lambda p: p.rstrip(), data_extraction)
data_json_str = "[" + ','.join(data_extraction) + "]"
df = pd.read_json(data_json_str)


dictionary = dict()
for i in range(0,N):
	for word in df['text'][i].split():
		if word not in dictionary and df['stars'][i] > 3:
			dictionary[word] = [1]
			dictionary[word].append(0)
		elif word not in dictionary and df['stars'][i] <= 3:
			dictionary[word] = [0]
			dictionary[word].append(1)
		elif word in dictionary and df['stars'][i] > 3:
			dictionary[word][0] +=1
		elif word in dictionary and df['stars'][i] <= 3:		
			dictionary[word][1] +=1
		else:
			pass


for k,v in dictionary.items():
	if (dictionary[k][0] < 0.2 * dictionary[k][1] and dictionary[k][1] > 10) or (dictionary[k][1] < 0.2 * dictionary[k][0] and dictionary[k][0] > 10):
		if len(k) > 2:
			bag_of_words.append(k)
	


print len(bag_of_words)

data = df['text']
rev = df['stars'] 



def product_helper(args):
	return featureExtraction(*args)


def featureExtraction(p,t):		
	temp = [0] * len(bag_of_words)
	for word in p.split():
		if word in bag_of_words and len(word)>2:
			temp[bag_of_words.index(word)] += 1
	
	if sum(temp)!=0:
		if t > 3:
			return temp + [1,0]
		else:
			return temp + [0,1]
	else:
		pass


def calculateParallel(threads):	
	pool = multiprocessing.Pool(threads)
	result = []
	job_args = [(item_a, rev[i]) for i, item_a in enumerate(data)]
	l=pool.map_async(product_helper,job_args,callback=result.extend)
	l.wait()
	pool.close()
	pool.join()
	return result


 
temp_X = calculateParallel(12)
temp_X = [x for x in temp_X if x is not None]


numROWS = len(temp_X)
numCOLUMNS = len(temp_X[0])
train_size = 4 * len(temp_X) / 5


train_X = []
train_y= []
test_X = []
test_y = []
for i in range(0,train_size):
	train_X.append(temp_X[i][0:numCOLUMNS-2])
	train_y.append(temp_X[i][numCOLUMNS-2:numCOLUMNS])

for i in range(train_size,numROWS):
	test_X.append(temp_X[i][0:numCOLUMNS-2])
	test_y.append(temp_X[i][numCOLUMNS-2:numCOLUMNS])


train_X = np.array(train_X)
test_X = np.array(test_X)
train_y = np.array(train_y)
test_y = np.array(test_y)





datapoints = train_X.shape[0]
w = np.zeros((train_X.shape[1],numclasses))
p = train_X.shape[1]

print "Done computation"

probes = 0;
def softmax(w, x):
	w = w.reshape((p,numclasses))
	act = np.dot(x,w)
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
	grad = np.dot(x.T,sig_activation)
	grad /= float(datapoints)
	grad = grad.ravel()
	return grad

def classify(w, x):
	prob = softmax(w, x)
	probes = prob
	prob = np.argmax(prob, axis=1).squeeze()
	return prob

def neg_log_likelihood(w, x, y):
	sig_activation = softmax(w,x)
	temp = sig_activation*y
	temp = temp.sum(axis=1)
	neg_log = -np.mean(np.log(temp))
	return neg_log




ret = fmin_l_bfgs_b(neg_log_likelihood, w, fprime=gradient, args=(train_X,train_y))
res = ret[0].reshape((train_X.shape[1],numclasses))

out = classify(res, test_X)
count = 0
for i in range(0,len(out)):
	if out[i] == 0:
		if test_y[i][0] == 1:
			count+=1
			
	else:
		if test_y[i][1] == 1:
			count+=1


print "Accuracy is {}".format(100 * count/float(len(out)))









