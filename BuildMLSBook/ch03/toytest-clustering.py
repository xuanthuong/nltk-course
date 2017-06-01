from sklearn.feature_extraction.text import CountVectorizer
import os, sys
import scipy as sp
import math
import nltk.stem

# vectorizer = CountVectorizer(min_df = 1)
# vectorizer = CountVectorizer(min_df = 1, stop_words='english')
english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer): #inherite from CountVectorizer
	def build_analyzer(self): #override method: build_analyzer in CountVectorizer
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))		
vectorizer = StemmedCountVectorizer(min_df = 1, stop_words='english')
# endvectorizer

DIR = 'D:/Dropbox/SharedWorks/PythonML/LearnPython/BuildMLSBook/ch03/toytest' 
# DIR = '../toytest' #not work #must '/' instead of '\'

#print(os.listdir(DIR))
posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]
X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape
print(X_train)
print(X_train.toarray())
print("#samples: %d, #features: %d" % (num_samples, num_features))
print(vectorizer.get_feature_names())
new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
#print(new_post_vec)
#print(new_post_vec.toarray())

# scipy
def dist_raw(v1, v2):
	delta = v1 - v2
	return sp.linalg.norm(delta.toarray())
def dist_norm(v1, v2):
	v1_normalized = v1/sp.linalg.norm(v1.toarray())
	v2_normalized = v2/sp.linalg.norm(v2.toarray())
	delta = v1_normalized - v2_normalized
	return sp.linalg.norm(delta.toarray())
# endscipy

#best_doc = None
#best_dist = sys.maxint
#best_i = None
#for i in range(0, num_samples):
	#post = posts[i]
	#if post == new_post:
		#continue
	#post_vec = X_train.getrow(i)
	#d = dist_norm(post_vec, new_post_vec)
	#print "=== Post %i with dist = %.2f: %s" % (i + 1, d, post)
	#if d < best_dist:
		#best_dist = d
		#best_i = i + 1
		
#print("Best post is post %i with dist = %.2f" % (best_i, best_dist))
#print(X_train.getrow(3).toarray())
#print(X_train.getrow(4).toarray())

#print("Test calc tf-idf")
def tfidf(term, doc, docset):
	# tf = float(doc.count(term))/sum(w.count(term) for w in docset)
	tf = float(doc.count(term)) / sum(doc.count(w) for w in set(doc)) 
	#tf = float(doc.count(term))/len(doc)
	idf = math.log(float(len(docset))/(len([doc for doc in docset if term in doc])))
	return tf*idf



import numpy as np
import matplotlib.pyplot as plt

ydata = X_train.toarray()
n = 15
x = np.arange(1, 18, 1)

i = 0
for y in ydata:
	plt.scatter(x,y, label = 'Doc ' + str(i))
	i = i + 1
plt.title("Visualization of document clustering")
plt.xlabel("WORDS / FEATURES")
plt.ylabel("TF-IDF")
plt.xticks(x, vectorizer.get_feature_names())
plt.legend() #Must have this to show the legend (labelled before)
plt.show()

	

