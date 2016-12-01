import pandas as pd
import numpy as np
import scipy as sp
import tokenize
from scipy import optimize
from scipy.sparse import csr_matrix
import math
from multiprocessing.dummy import Pool as ThreadPool
from threading import Thread, current_thread
import multiprocessing
import heapq

N=1000
alpha = 0.7
epsilon = 0.00001
batchsize = 512
iterations = 100
no_of_cores = 24
no_of_classes = 5
train_size = 4 * N / 5
bag_of_words=[]


with open('test_yelp_academic_dataset_review.json', 'r') as file_req:
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
		if len(temp_word) <= 2:
			continue
		if temp_word not in dictionary:
			if df['stars'][i] == 1:
				dictionary[temp_word] = [1,0,0,0,0]
			elif df['stars'][i] == 2:
                                dictionary[temp_word] = [0,1,0,0,0]
			elif df['stars'][i] == 3:
                                dictionary[temp_word] = [0,0,1,0,0]
			elif df['stars'][i] == 4:
                                dictionary[temp_word] = [0,0,0,1,0]
			elif df['stars'][i] == 5:
                                dictionary[temp_word] = [0,0,0,0,1]
		elif temp_word in dictionary:
			if df['stars'][i] == 1:
				dictionary[temp_word][0] +=1
                        if df['stars'][i] == 2:
                                dictionary[temp_word][1] +=1
                        if df['stars'][i] == 3:
                                dictionary[temp_word][2] +=1
                        if df['stars'][i] == 4:
                                dictionary[temp_word][3] +=1
                        if df['stars'][i] == 5:
                                dictionary[temp_word][4] +=1
		else:
			pass

frequencies = [0]
for k,v in dictionary.items():
	frequencies.append(sum(dictionary[k]))

frequencies = np.array(frequencies)
maxoccurrence = min(len(dictionary.keys()), 10000)

threshold = heapq.nlargest(maxoccurrence, frequencies)[maxoccurrence-1]

for k,v in dictionary.items():
	if sum(dictionary[k]) >= threshold:
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
		if temp_word in bag_of_words and len(temp_word)>2:
			temp[bag_of_words.index(temp_word)] += 1
	
	if sum(temp)!=0:
		if t == 1:
			return temp + [1,1,0,0,0,0]
		elif t == 2:
			return temp + [1,0,1,0,0,0]
		elif t == 3:
			return temp + [1,0,0,1,0,0]
		elif t == 4:
			return temp + [1,0,0,0,1,0]
		elif t == 5:
			return temp + [1,0,0,0,0,1]

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
	train_X.append(temp_X[i][0:numCOLUMNS-no_of_classes])
	train_y.append(temp_X[i][numCOLUMNS-no_of_classes:numCOLUMNS])

for i in range(train_size,numROWS):
	test_X.append(temp_X[i][0:numCOLUMNS-no_of_classes])
	test_y.append(temp_X[i][numCOLUMNS-no_of_classes:numCOLUMNS])


train_X = np.array(train_X)
test_X = np.array(test_X)
train_y = np.array(train_y)
test_y = np.array(test_y)

datapoints = train_X.shape[0]
w = np.zeros((train_X.shape[1],no_of_classes))

print "Done computation", w.shape

def softmax(w, x):
	w = w.reshape((train_X.shape[1],no_of_classes))
	act = x.dot(w)
	act -= act.max(axis=1)[:, np.newaxis]
	exp_act = np.exp(act)
	fin_res = exp_act/exp_act.sum(axis=1)[:,np.newaxis]
	return fin_res


def gradient(w,x,y):
	sig_activation = softmax(w,x);
	temp = sig_activation * y
	temp = temp/temp.sum(axis=1)[:,np.newaxis]
	sig_activation -= temp
	grad = np.dot(x.T, sig_activation)
	grad /= float(datapoints)
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

def stoc_grad_desc(x, y):
    w = np.zeros((train_X.shape[1],no_of_classes), dtype=np.uint32)
    nll_prev = neg_log_likelihood(w, x, y)
    for i in range(0,iterations):
        k = 0
        while k < len(x):
	    end = min(k+batchsize, len(x))
            w = w - (alpha * gradient(w, x[k:end,:], y[k:end, :]))
            k = k + batchsize

        nll = neg_log_likelihood(w, x, y) 
        diff = np.abs(nll - nll_prev)
        nll_prev = nll;
        if diff < epsilon :
            print diff
            print "iteration no : ", i
            break

    print "iteration no : ", i, " nll ", nll_prev
    return w

res = stoc_grad_desc(train_X, train_y)
out = classify(res, test_X)
print out

confusion_matrix = [[0 for i in range(5)] for j in range(5)]
for i in range(0, len(test_y)):
	original_class = np.argmax(test_y[i]).squeeze()
	confusion_matrix[out[i]][original_class]+=1


accuracy = 0
for j in range(0,len(confusion_matrix)):
        accuracy = accuracy + confusion_matrix[j][j]                                    
accuracy = (100 * accuracy) / float(len(test_y))                                       
                
                
precision = [0,0,0,0,0]
for j in range(0,len(confusion_matrix)):                                                
        if sum(confusion_matrix[j][:]) == 0:                                            
                precision[j] = 0
        else:                                                                           
                precision[j] =  (100 * confusion_matrix[j][j]) / float(sum(confusion_matrix[j][:]))                                                                              
                                                                                        
recall = [0,0,0,0,0]
for j in range(0,len(confusion_matrix)):
        if sum(row[j] for row in confusion_matrix) == 0:                                
                recall[j] = 0                                                           
        else:
                recall[j] =     (100 * confusion_matrix[j][j]) / float(sum(row[j] for row in confusion_matrix))


print "Accuracy is {}".format(accuracy)
print "Precision = {}".format(precision)
print "Recall = {}".format(recall)

