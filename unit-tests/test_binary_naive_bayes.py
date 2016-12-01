import pandas as pd
import numpy as np
from scipy.optimize import fmin_l_bfgs_b	
import math
from multiprocessing.dummy import Pool as ThreadPool
from threading import Thread, current_thread
import multiprocessing
import sys
sys.path.append("/usr/local/lib/python2.7/site-packages")
#from nltk import WordNetLemmatizer
#lmtzr = WordNetLemmatizer()

N=1000
dictionary=[]
numclasses = 2
with open('test_yelp_academic_dataset_review.json', 'r') as file_req:
	data_extraction = file_req.readlines()[0:N]
		
data_extraction = map(lambda p: p.rstrip(), data_extraction)
data_json_str = "[" + ','.join(data_extraction) + "]"
df = pd.read_json(data_json_str)



pos_review_count=0
neg_review_count=0
pos_word_count = 0 
neg_word_count = 0  

train_size = 4*N/5

dictionary = dict()
for i in range(0,train_size):
	if df['stars'][i] > 3:
		pos_review_count +=1
	else:
		neg_review_count +=1	
	for word in df['text'][i].split():
		temp_word = word.lower()
#		temp_word = lmtzr.lemmatize(temp_word)
		if df['stars'][i] > 3:
			pos_word_count+=1
			if temp_word not in dictionary:
				dictionary[temp_word] = [1,0]
			else:
				dictionary[temp_word][0] +=1	
		elif df['stars'][i] < 3:
			neg_word_count+=1
			if temp_word not in dictionary:
				dictionary[temp_word] = [0,1]
			else:
				dictionary[temp_word][1] +=1



prob_pos = pos_review_count / float(pos_review_count + neg_review_count)
prob_neg = neg_review_count / float(pos_review_count + neg_review_count)

prob_pos = math.log(prob_pos,2)
prob_neg = math.log(prob_neg,2)
print len(dictionary)

temp_pos = math.log(pos_word_count+len(dictionary),2)
temp_neg = math.log(neg_word_count+len(dictionary),2)


confusion_matrix = [[0,0],[0,0]]
print "Started testing"
for i in range(train_size,N):
	count=0
	final_prob_pos = prob_pos
	final_prob_neg = prob_neg
	for word in df['text'][i].split():	
		temp_word = word.lower()
		count+=1
#		temp_word = lmtzr.lemmatize(temp_word)
		if temp_word in dictionary:			
			final_prob_pos = final_prob_pos + math.log((dictionary[temp_word][0]+1),2)			
			final_prob_neg = final_prob_neg + math.log((dictionary[temp_word][1]+1),2)
			
	final_prob_pos = final_prob_pos - count * temp_pos
	final_prob_neg = final_prob_neg - count * temp_neg
	if df['stars'][i] > 3:
		if final_prob_pos > final_prob_neg:
			confusion_matrix[0][0]+=1
		else:
			confusion_matrix[1][0]+=1
		continue
	else:
		if final_prob_pos < final_prob_neg:
			confusion_matrix[1][1]+=1
		else:
			confusion_matrix[0][1]+=1
		continue	
		

accuracy = (100 * (confusion_matrix[0][0] + confusion_matrix[1][1])) / float(N-train_size)
precision1 = (100 * confusion_matrix[0][0]) / float(confusion_matrix[0][0] + confusion_matrix[0][1])
precision2 = (100 * confusion_matrix[1][1]) / float(confusion_matrix[1][0] + confusion_matrix[1][1])
recall1 = (100 * confusion_matrix[0][0]) / float(confusion_matrix[0][0] + confusion_matrix[1][0])
recall2 = (100 * confusion_matrix[1][1]) / float(confusion_matrix[0][1] + confusion_matrix[1][1])	

print "Accuracy is {}".format(accuracy)
print "Precision of label 1 is {}".format(precision1)
print "Recall of label 1 is {}".format(recall1)
print "Precision of label 2 is {}".format(precision2)
print "Recall of label 2 is {}".format(recall2)
	
		
		

