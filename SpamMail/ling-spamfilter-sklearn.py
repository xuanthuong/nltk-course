#  -*- coding: utf-8 -*-
"""
@blog: http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html

Practiced on Jun 02, 2017
@author: Thuong Tran
@Library: scikit-learn
"""

import os, glob, random
import numpy as np
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
import time


SPAM = 'spam'
HAM = 'ham'


def build_data_frame(mail_dir):  
  email_files = glob.glob(mail_dir)
  print('Number of email files: %s' % len(email_files))
  
  rows = []
  index = []

  for mail in email_files:
    with open(mail) as m:
      for i, line in enumerate(m):
        if i == 2:
          if 'spmsg' in os.path.basename(mail):
            rows.append({'text': line, 'class': SPAM})
            index.append(mail)
          else:
            rows.append({'text': line, 'class': HAM})
            index.append(mail)
  data_frame = DataFrame(rows, index = index)
  data_frame = data_frame.reindex(np.random.permutation(data_frame.index))
  return data_frame 


def main():
  # Read raw emails and build datasets
  mails_dir = '/Users/thuong/Documents/tmp_datasets/SpamMail/lingspam_public/lemm_stop/*/*.txt'
  data_frame = build_data_frame(mails_dir)
  count_vectorizer = CountVectorizer()
  print('Data frame shape: %s, rows: %s' % (data_frame.shape, len(data_frame)))

  # Feature extraction
  print ('Starting feature extraction...')
  start_time = time.time()
  counts = count_vectorizer.fit_transform(data_frame['text'].values)
  targets = data_frame['class'].values
  print ('Feature extraction time: %s' % (time.time() - start_time))

  size = int(len(data_frame) * 0.7)
  print('size: %s' % size)
  train_data, test_data = counts[:size], counts[size:]
  train_target, test_target = targets[:size], targets[size:]

  # Training
  print ('Start training...')
  start_time = time.time()
  nb_classifier = MultinomialNB()
  svc_classifier = LinearSVC()  
  nb_classifier.fit(train_data, train_target)
  svc_classifier.fit(train_data, train_target)
  print('training time: %s minutes' % (time.time() - start_time))

  # Testing and evaluation
  result_of_svm = svc_classifier.predict(test_data)
  result_of_nb = nb_classifier.predict(test_data)

  print (confusion_matrix(test_target, result_of_svm))
  print ("SMV Score: %s" % f1_score(test_target, result_of_svm, pos_label = SPAM))
  print (confusion_matrix(test_target, result_of_nb))
  print ("Navie Bayes Score: %s" % f1_score(test_target, result_of_nb, pos_label = SPAM))


def pipeline_main():
  mails_dir = '/Users/thuong/Documents/tmp_datasets/SpamMail/lingspam_public/lemm_stop/*/*.txt'
  data_frame = build_data_frame(mails_dir)

  pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())])

  size = int(len(data_frame) * 0.7)
  train_data, test_data = data_frame[:size]['text'].values, data_frame[size:]['text'].values
  train_target, test_target = data_frame[:size]['class'].values, data_frame[size:]['class'].values
  pipeline.fit(train_data, train_target)
  predictions = pipeline.predict(test_data)

  print('Confusion matrix with one-fold: ')
  print(confusion_matrix(test_target, predictions))
  print("Score with one-fold: %s" % f1_score(test_target, predictions, pos_label = SPAM))

  k_fold = KFold(n=len(data_frame), n_folds=6)
  scores = []
  confusion = np.array([[0, 0], [0, 0]])
  for train_indices, test_indices in k_fold:
    train_text = data_frame.iloc[train_indices]['text'].values
    train_label = data_frame.iloc[train_indices]['class'].values
    test_text = data_frame.iloc[test_indices]['text'].values
    test_label = data_frame.iloc[test_indices]['class'].values

    pipeline.fit(train_text, train_label)
    predictions = pipeline.predict(test_text)

    confusion += confusion_matrix(test_label, predictions)
    score = f1_score(test_label, predictions, pos_label = SPAM)
    scores.append(score)

  print('Confusion matrix with 6-fold: ')
  print(confusion)
  print('Score with 6-fold: %s' % (sum(scores)/len(scores)))


if __name__ == "__main__":
  # main()
  pipeline_main()
