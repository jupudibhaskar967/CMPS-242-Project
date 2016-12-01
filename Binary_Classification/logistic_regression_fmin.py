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
from multiprocessing.dummy import Pool as ThreadPool
from threading import Thread, current_thread
import multiprocessing


N=10000
no_of_cores = 24
lam = 0.00001
train_size = 4 * N / 5
bag_of_words=[]

with open('../yelp_academic_dataset_review.json', 'r') as file_req:
	data_extraction = file_req.readlines()[0:N]

stop_words = ['d', 'theirs', 'ourselves', 'no', 'your', 'nor', 'other', 'off', 'very', 'from', 'now', 'only', 'between', 'too', 'having', 'm', 'y', 'myself', 'did', 'am', 'those', 'does', 'own', 'if', 'then', 'here', 'same', 't', 'our', 'wasn', 'until', 'you', 'below', 'once', 'an', 'ain', 'the', 'being', 'himself', 'more', 'didn', 'themselves', 'or', 'a', 'which', 'few', 'some', 'to', 'through', 'out', 'over', 'of', 'up', 'isn', 'aren', 'mightn', 'we', 'll', 'yourself', 'it', 'so', 'my', 'against', 'by', 'itself', 'this', 'ours', 'again', 'that', 'while', 'do', 'his', 'not', 'but', 'she', 're', 'can', 'with', 'about', 'haven', 'me', 'hadn', 'shouldn', 'before', 'hasn', 'in', 'been', 'who', 'her', 'all', 'there', 'after', 'most', 'their', 'had', 'i', 'than', 'doesn', 'down', 'be', 'him', 'shan', 'whom', 'don', 'will', 'needn', 'won', 'why', 'how', 'have', 'are', 'doing', 'further', 'were', 'ma', 'such', 'herself', 'these', 'hers', 'o', 'under', 'and', 'both', 'he', 'where', 'at', 'above', 's', 'they', 'is', 've', 'wouldn', 'its', 'any', 'yourselves', 'because', 'weren', 'what', 'just', 'them', 'for', 'on', 'as', 'should', 'each', 'during', 'couldn', 'was', 'mustn', 'when', 'into', 'yours', 'has']

stop_words_dict = dict()
for i in range(0, len(stop_words)):
	stop_words_dict[stop_words[i]] = 1

		
data_extraction = map(lambda p: p.rstrip(), data_extraction)
data_json_str = "[" + ','.join(data_extraction) + "]"
df = pd.read_json(data_json_str)

dictionary = dict()
for i in range(0, train_size):
	for word in df['text'][i].split():
		temp_word = word.lower()
		if temp_word in stop_words_dict:
			continue
		if temp_word not in dictionary and df['stars'][i] > 3:
			dictionary[temp_word] = [1,0]
		elif temp_word not in dictionary and df['stars'][i] < 3:
			dictionary[temp_word] = [0,1]
		elif temp_word in dictionary and df['stars'][i] > 3:
			dictionary[temp_word][0] +=1
		elif temp_word in dictionary and df['stars'][i] < 3:		
			dictionary[temp_word][1] +=1
		else:
			pass


for k,v in dictionary.items():
	if ((dictionary[k][0] < 0.4 * dictionary[k][1]) and dictionary[k][1]>14) or ((dictionary[k][1] < 0.4 * dictionary[k][0]) and dictionary[k][0]>14):
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
		temp_word = word.lower()
		#temp_word = lmtzr.lemmatize(temp_word)
		if temp_word in bag_of_words and len(temp_word)>2:
			temp[bag_of_words.index(temp_word)] += 1
	
	if sum(temp)!=0:
		if t > 3:
			return temp + [1,1]
		elif t < 3:
			return temp + [1,0]
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


 
temp_X = calculateParallel(no_of_cores)
temp_X = [x for x in temp_X if x is not None]


numROWS = len(temp_X)
numCOLUMNS = len(temp_X[0])
train_size = 4 * len(temp_X) / 5


train_X = []
train_y= []
test_X = []
test_y = []
for i in range(0,train_size):
	train_X.append(temp_X[i][0:numCOLUMNS-1])
	train_y.append(temp_X[i][numCOLUMNS-1:numCOLUMNS])

for i in range(train_size,numROWS):
	test_X.append(temp_X[i][0:numCOLUMNS-1])
	test_y.append(temp_X[i][numCOLUMNS-1:numCOLUMNS])


train_X = np.array(train_X)
test_X = np.array(test_X)
train_y = np.array(train_y)
test_y = np.array(test_y)

datapoints = train_X.shape[0]
w = np.zeros((train_X.shape[1],1))

print "Done computation", w.shape

def sigmoid_func(w, x):
	w = w.reshape((train_X.shape[1],1))
	act = -np.dot(x, w)
	exp_act = np.exp(act)
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


ret = optimize.fmin_l_bfgs_b(neg_log_likelihood, np.zeros((train_X.shape[1],1)), fprime=gradient, args=(train_X,train_y))

res = ret[0].reshape((train_X.shape[1],1))

out = classify(res, test_X)
print np.transpose(out)


confusion_matrix = [[0,0],[0,0]]
for i in range(0, len(test_y)):
	if test_y[i] == 1:
		if out[i] == 1:
			confusion_matrix[0][0]+=1
		else:
			confusion_matrix[1][0]+=1
	else:
		if out[i] == 0:
			confusion_matrix[1][1]+=1
		else:
			confusion_matrix[0][1]+=1

#print " accuracy is ", (count/float(len(test_y)))*100

accuracy = (100 * (confusion_matrix[0][0] + confusion_matrix[1][1])) / float(len(test_y))
precision1 = (100 * confusion_matrix[0][0]) / float(confusion_matrix[0][0] + confusion_matrix[0][1])
precision2 = (100 * confusion_matrix[1][1]) / float(confusion_matrix[1][0] + confusion_matrix[1][1])
recall1 = (100 * confusion_matrix[0][0]) / float(confusion_matrix[0][0] + confusion_matrix[1][0])
recall2 = (100 * confusion_matrix[1][1]) / float(confusion_matrix[0][1] + confusion_matrix[1][1])       

print "Accuracy is {}".format(accuracy)
print "Precision for positive reviews is {}".format(precision1)
print "Recall for positive reviews is {}".format(recall1)
print "Precision for negative reviews is {}".format(precision2)
print "Recall for negative reviews is {}".format(recall2)
