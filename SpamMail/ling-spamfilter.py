#  -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 22:53:50 2017
@author: Abhijeet Singh
@blog: https://appliedmachinelearning.wordpress.com

Practiced on May 31, 2017
@author: Thuong Tran
@Library: scikit-learn
"""

import os, glob, random
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import time


def read_all_mails(mail_dir):  
  email_files = glob.glob(mail_dir)
  print('Number of email files: %s' % len(email_files))
  ham_mails = []
  ham_labels = []
  spam_mails = []
  spam_labels = []
  ham_mails_labels = []
  spam_mails_labels = []

  train_mails = []
  train_labels = []
  test_mails = []
  test_labels = []
  for mail in email_files:
    with open(mail) as m:
      for i, line in enumerate(m): # i: index, line: value
        if i == 2:
          if 'spmsg' in os.path.basename(mail):
            spam_mails_labels.append((line, 1))
          else:
            ham_mails_labels.append((line, 0))
  random.shuffle(spam_mails_labels)
  random.shuffle(ham_mails_labels)

  for mail, label in ham_mails_labels:
    ham_mails.append(mail)
    ham_labels.append(label)
  for mail, label in spam_mails_labels:
    spam_mails.append(mail)
    spam_labels.append(label)
  
  test_size = 130
  train_mails = ham_mails[test_size:] + spam_mails[test_size:]
  train_labels = ham_labels[test_size:] + spam_labels[test_size:]

  test_mails = ham_mails[:test_size] + spam_mails[:test_size]
  test_labels = ham_labels[:test_size] + spam_labels[:test_size]

  all_mails = train_mails + test_mails

  return all_mails, train_mails, train_labels, test_mails, test_labels


def make_dictionary(all_mails):  
  all_words = []
  for mail in all_mails:  
    words = mail.split()
    all_words += words

  dictionary = Counter(all_words) # return dictionary: key: word, value: number of occurence

  list_to_remove = dictionary.keys()
  for item in list_to_remove:
    if item.isalpha() == False: # False if word has at least 1 number or white space, otherwise: True
      del dictionary[item]
    elif len(item) == 1:
      del dictionary[item] # word has only 1 character
  dictionary = dictionary.most_common(3000) # only put 3000 common words into dictionary
  return dictionary


def extract_features(dictionary, all_mails):  
  num_mails = len(all_mails)
  features_vector = np.zeros((num_mails, 3000)) # sparse matrix
  docID = 0
  for mail in all_mails:    
    words = mail.split()
    for word in words:
      wordID = 0
      for j, d in enumerate(dictionary):
        if d[0] == word:
          wordID = j
          features_vector[docID, wordID] = words.count(word)
    docID += 1
  return features_vector


def main():
  # Create a dictionary of words with its frequency
  train_dir = '/Users/thuong/Documents/tmp_datasets/SpamMail/lingspam_public/lemm_stop/*/*.txt'
  all_mails, train_mails, train_labels, test_mails, test_labels = read_all_mails(train_dir)
  print('Numer of mails: %s' % len(all_mails))
  print('Number of train emails: ', len(train_mails))
  print('Number of test mails: ', len(test_mails))
  dictionary = make_dictionary(all_mails)
  # print(dictionary[0:10])

  # Prepare feature vectors per training mail and its labels
  print ('Start feature extraction...')
  start_time = time.time()
  train_matrix = extract_features(dictionary, train_mails)
  print('Feature extraction elapsed time is: %s minutes' % (time.time() - start_time))

  # Training SVM and Naive bayes classifier and its variants
  print ('Start training...')
  model_svm = LinearSVC()
  model_nb = MultinomialNB()
  start_time = time.time()
  model_svm.fit(train_matrix, train_labels)
  model_nb.fit(train_matrix, train_labels)
  print('training time: %s minutes', (time.time() - start_time))

  # Test the unseen mails for Spam
  test_matrix = extract_features(dictionary, test_mails)
  result_of_svm = model_svm.predict(test_matrix)
  result_of_nb = model_nb.predict(test_matrix)

  print (confusion_matrix(test_labels, result_of_svm))
  print (confusion_matrix(test_labels, result_of_nb))

if __name__ == "__main__":
  main()
