import pandas as pd
import operator
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import tokenize
from scipy.optimize import fmin_l_bfgs_b
import math
from scipy.sparse import csr_matrix
import timeit
from collections import _count_elements
import time
import string
import threading
import memory_profiler as profiler
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer


N = 100000
bag_of_words = []
numclasses = 2

JSON_open = time.clock()
with open('yelp_academic_dataset_review.json', 'r') as file_req:
    data_extraction = file_req.readlines()[0:N]
file_req.close()
JSON_close = time.clock()

print("Time to read JSON File : " + str(JSON_close - JSON_open))
print("Number of reviews being used : " + str(N))

#data_extraction = map(lambda p: p.strip(' ,:;!@#$%^&*()123456789+=?."'), data_extraction)

data_extraction = map(lambda p: p.strip(), data_extraction)
data_json_str = "[" + ','.join(data_extraction) + "]"
df = pd.read_json(data_json_str)


translator = str.maketrans({key: None for key in string.punctuation + '0123456789'})
good_dict = dict()
bad_dict = dict()
cachedStopWords = set(stopwords.words('english'))

rev_list = [df['text'][i] for i in range(0,N)]
rat_list = [df['stars'][i] for i in range(0,N)]
create_list = time.clock()
lmtzr = WordNetLemmatizer()
print("Time to create review and rating list : " + str(create_list-JSON_close))

good_rev_list = [rev_list[i].translate(translator).split() for i in range(0,N) if(rat_list[i]>3)]
bad_rev_list = [rev_list[i].translate(translator).split() for i in range(0,N) if(rat_list[i]<3)]
good_rev_list = [[lmtzr.lemmatize(word.lower()) for word in good_rev]for good_rev in good_rev_list]
bad_rev_list = [[lmtzr.lemmatize(word.lower()) for word in bad_rev]for bad_rev in bad_rev_list]

all_rev_list = good_rev_list + bad_rev_list
good_bad_time = time.clock()

print("Time to sort good & bad reviews : " + str(good_bad_time-create_list))

print("the length of good rev list: " + str(len(good_rev_list)))
print("the length of bad rev list : " + str(len(bad_rev_list)))


start_good_dict = time.clock()

for i in range(0,len(good_rev_list)):
    for word in good_rev_list[i]:
        if word not in good_dict and word not in cachedStopWords and len(word)>2 :
            good_dict[word] = 1
        elif word in good_dict:
            good_dict[word] += 1
end_good_dict = time.clock()


for i in range(0,len(bad_rev_list)):
    for word in bad_rev_list[i]:
        if word not in bad_dict and word not in cachedStopWords and len(word)>2 :
            bad_dict[word] = 1
        elif word in good_dict:
            bad_dict[word] += 1

end_bad_dict = time.clock()

##remove words from training set if they occur only once
good_dict = {key:value for key,value in good_dict.items() if value>1}
bad_dict =  {key:value for key,value in bad_dict.items() if value>1}


print("Time to build good dict : " + str(end_good_dict-start_good_dict))
print("Time to build bad dict : " + str(end_bad_dict-end_good_dict))

print("Number of good words : " + str(len(good_dict)))
print("Number of bad words : " + str(len(bad_dict)))


inter_set = set(good_dict).intersection(set(bad_dict))
print("Length of intersection set : " + str(len(inter_set)))

## Remove words which occur in both good and bad reviews and also have similar count
start_bag_of_words = time.clock()
for word in inter_set:
    if (abs(good_dict[word]-bad_dict[word])/min(good_dict[word],bad_dict[word]))<0.5:
        del good_dict[word],bad_dict[word]
end_bag_of_words = time.clock()



print("Time to generate bag of words : " + str(end_bag_of_words-start_bag_of_words))
print("After Deleteee of repeat words with similar count :::::::")
print("Number of good words : " + str(len(good_dict)))
print("Number of bad words : " + str(len(bad_dict)))

BOW_dict = {**good_dict, **bad_dict}
print("Number of words in BOW Dict :" + str(len(BOW_dict)))

BOW_list = list(BOW_dict.keys())
print("Number of words in BOW List :" + str(len(BOW_list)))


'''
###
print("Building count vectorssss!!!!!!!")

start_count_vec = time.clock()
count_vec = []
for rev in all_rev_list:
    temp_count_vec = [0] * len(BOW_list)
    for word in rev:
        if word in BOW_list:
            temp_count_vec[BOW_list.index(word)] += 1
    count_vec.append(temp_count_vec)
end_count_vec = time.clock()
print("length of count vector is " + str(len(count_vec)))
print("time to generate count vector is : " + str(end_count_vec-start_count_vec))
print("Bag of words hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
for i in list(BOW_dict.keys()):
    print(i)


#print(count_vec)


## Print top words from good and bad dictionary's
sorted_good_dict = sorted(good_dict.items(),key=operator.itemgetter(1),reverse = True)
sorted_bad_dict = sorted(bad_dict.items(),key=operator.itemgetter(1), reverse = True)
print("Good words hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
for i in sorted_good_dict:
    print(i[0])

print("Bad words hereeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
for i in sorted_bad_dict:
    print(i[0])




"""


#print(rev_list)





























@profiler.profile
def fun():
    start = time.clock()
    N = 1200
    bag_of_words = []
    numclasses = 2
    with open('yelp_academic_dataset_review.json', 'r') as file_req:
        data_extraction = file_req.readlines()[0:N]
        file_req.close()
    end = time.clock()
    print(len(data_extraction))
    print( "The run  time is : " + str(end - start))
    return


if __name__ == '__main__':
    fun()

"""