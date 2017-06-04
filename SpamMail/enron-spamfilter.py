#  -*- coding: utf-8 -*-
"""
@blog: http://www.cs.ucf.edu/courses/cap5636/fall2011/nltk.pdf

Practiced on Jun 02, 2017
@author: Thuong Tran
@Library: nltk
"""
from nltk import word_tokenize, NaiveBayesClassifier
from nltk import classify
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import random
import sys, os, glob, re
import codecs

word_lemmatizer = WordNetLemmatizer()
common_words = stopwords.words('english')
# common_words = stem.snowball.EnglishStemmer().stem


def main():
  # reload(sys)  
  # sys.setdefaultencoding('utf8')
  data_dir = '/Users/thuong/Documents/tmp_datasets/SpamMail/enron2006/*'
  hamtexts = []
  spamtexts = []

  for filename in glob.glob(os.path.join(data_dir, 'ham/*.txt')):
    fin = codecs.open(filename, encoding = 'latin1')
    hamtexts.append(fin.read())
    fin.close()
  print('Number of ham emails are: %s' % len(hamtexts))

  for filename in glob.glob(os.path.join(data_dir, 'spam/*.txt')):
    fin = codecs.open(filename, encoding = 'latin1')
    spamtexts.append(fin.read())
    fin.close()
  print('Number of spam emails are: %s' % len(spamtexts))
  print('Total number of emails are: %s' % (len(hamtexts) + len(spamtexts)))

  mixed_emails = [(email, 'spam') for email in spamtexts]
  mixed_emails += [(email, 'ham') for email in hamtexts]
  random.shuffle(mixed_emails)

  feature_sets = [(feature_extractor(email), label) \
                  for (email, label) in mixed_emails]
  # Training and Testing
  size = int(len(feature_sets) * 0.7)
  train_set, test_set = feature_sets[size:], feature_sets[:size]

  print ('train set size = %d, test set size = %d' \
          % (len(train_set), len(test_set)))
  classifier = NaiveBayesClassifier.train(train_set)

  print (classify.accuracy(classifier, test_set))
  classifier.show_most_informative_features(20)


def feature_extractor(sent):
  features = {}
  word_tokens = [word_lemmatizer.lemmatize(word.lower()) \
                  for word in word_tokenize(sent)]

  for word in word_tokens:
    if word not in common_words:
      features[word] = True
  
  return features


if __name__ == "__main__":
  main()