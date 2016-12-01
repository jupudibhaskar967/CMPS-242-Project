import pandas as pd
import numpy as np
from scipy.optimize import fmin_l_bfgs_b	
import math
from multiprocessing.dummy import Pool as ThreadPool
from threading import Thread, current_thread
import multiprocessing
import sys
import operator
sys.path.append("/usr/local/lib/python2.7/site-packages")

N=1000
dictionary=[]
with open('test_yelp_academic_dataset_review.json', 'r') as file_req:
	data_extraction = file_req.readlines()[0:N]
		
data_extraction = map(lambda p: p.rstrip(), data_extraction)
data_json_str = "[" + ','.join(data_extraction) + "]"
df = pd.read_json(data_json_str)

review_count = [0,0,0,0,0]
word_count = [0,0,0,0,0]


train_size = 4*N/5

dictionary = dict()
for i in range(0,train_size):
	if df['stars'][i] == 5:
		review_count[4]+=1
	elif df['stars'][i] == 4:
		review_count[3]+=1
	elif df['stars'][i] == 3:
		review_count[2]+=1
	elif df['stars'][i] == 2:
		review_count[1]+=1
	elif df['stars'][i] == 1:
		review_count[0]+=1
	else:
		pass					
	
	for word in df['text'][i].split():
		temp_word = word.lower()
		if df['stars'][i] == 1:
			word_count[0]+=1
			if temp_word not in dictionary:
				dictionary[temp_word] = [1,0,0,0,0]
			else:
				dictionary[temp_word][0] +=1	
		
		elif df['stars'][i] == 2:
			word_count[1]+=1
			if temp_word not in dictionary:
				dictionary[temp_word] = [0,1,0,0,0]
			else:
				dictionary[temp_word][1] +=1
		
		elif df['stars'][i] == 3:
			word_count[2]+=1
			if temp_word not in dictionary:
				dictionary[temp_word] = [0,0,1,0,0]
			else:
				dictionary[temp_word][2] +=1
		
		elif df['stars'][i] == 4:
			word_count[3]+=1
			if temp_word not in dictionary:
				dictionary[temp_word] = [0,0,0,1,0]
			else:
				dictionary[temp_word][3] +=1
		elif df['stars'][i] == 5:
			word_count[4]+=1
			if temp_word not in dictionary:
				dictionary[temp_word] = [0,0,0,0,1]
			else:
				dictionary[temp_word][4] +=1		
		else:
			pass


prob = [0,0,0,0,0]
for i in range(0,len(prob)):
	prob[i] = review_count[i] / float(sum(review_count))
	
print "dictionary length ",len(dictionary)
print "Started testing"
final_prob = [0,0,0,0,0]
confusion_matrix = [[0 for i in range(5)] for j in range(5)]
for i in range(train_size,N):
	count=0	
	for j in range(0,len(final_prob)):
		final_prob[j] = math.log(prob[j],2)		
	for word in df['text'][i].split():	
		temp_word = word.lower()
		count+=1
		if temp_word in dictionary:
			for j in range(0,len(final_prob)):			
				final_prob[j] = final_prob[j] + math.log((dictionary[temp_word][j]+1),2)
			
	
	for j in range(0,len(final_prob)):
		final_prob[j] = final_prob[j] - count * math.log(word_count[j]+len(dictionary),2)
	
	
	
	index, value = max(enumerate(final_prob), key=operator.itemgetter(1))
	if df['stars'][i]-1 == index:
		confusion_matrix[index][index]+=1
	else:
		confusion_matrix[index][df['stars'][i]-1]+=1		
		
				
accuracy = 0
for j in range(0,len(confusion_matrix)):
	accuracy = accuracy + confusion_matrix[j][j]	
accuracy = (100 * accuracy) / float(N-train_size)	
		
		
precision = [0,0,0,0,0]
for j in range(0,len(confusion_matrix)):
	if sum(confusion_matrix[j][:]) == 0:
		precision[j] = 0
	else:	
		precision[j] = 	(100 * confusion_matrix[j][j]) / float(sum(confusion_matrix[j][:]))

	  	

recall = [0,0,0,0,0]
for j in range(0,len(confusion_matrix)):
	if sum(row[j] for row in confusion_matrix) == 0:
		recall[j] = 0
	else:	
		recall[j] = 	(100 * confusion_matrix[j][j]) / float(sum(row[j] for row in confusion_matrix))	
		

print "Accuracy is {}".format(accuracy)
print "Precision = {}".format(precision)
print "Recall = {}".format(recall)
