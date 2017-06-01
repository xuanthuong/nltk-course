import sklearn.datasets
import scipy as sp
MLCOMP_DIR = "D:/dataset" # r"D:" and "D:" are same
data = sklearn.datasets.load_mlcomp("20news-18828", mlcomp_root = MLCOMP_DIR)
# all_data = sklearn.datasets.fetch_20newsgroups(subset="all")
# print("Number of total posts: %i" % len(all_data.filenames))
print(data.filenames)
print(len(data.filenames))
print(data.target_names)
print("-------")
train_data = sklearn.datasets.load_mlcomp("20news-18828", "train", mlcomp_root=MLCOMP_DIR)
print(len(train_data.filenames))
test_data = sklearn.datasets.load_mlcomp("20news-18828", "test", mlcomp_root=MLCOMP_DIR)
print(len(test_data.filenames))

print("----Simplicity---")
groups = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'sci.space']
train_data = sklearn.datasets.load_mlcomp("20news-18828", "train", mlcomp_root=MLCOMP_DIR, categories = groups)
print(len(train_data.filenames))

import nltk.stem
english_stemmer = nltk.stem.SnowballStemmer('english')
from sklearn.feature_extraction.text import TfidfVectorizer

class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(TfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=10, max_df=0.5,
                                    stop_words='english', decode_error='ignore'
                                    )

vectorized = vectorizer.fit_transform(train_data.data)
num_samples, num_features = vectorized.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))

labels = train_data.target
print("number of cluster = %i" % sp.unique(labels).shape[0])
num_clusters = 50  # sp.unique(labels).shape[0]
from sklearn.cluster import KMeans
km = KMeans(n_clusters = num_clusters, init = 'random', n_init = 1, verbose = 1)
# km = KMeans(n_clusters=num_clusters, n_init=1, verbose=1, random_state=3)
km.fit(vectorized)
print(km.labels_)
print(km.labels_.shape)

print("End training process")
print("Start to predict new incomming post")
#new_post = 'Disk drive problems. Hi, I have a problem with my hard disk. After 1 year it is working only sporadically now. I tried to format it, but now it doesn\'\t boot any more. Any ideas? Thanks.'
new_post = \
    """Disk drive problems. Hi, I have a problem with my hard disk.
After 1 year it is working only sporadically now.
I tried to format it, but now it doesn't boot any more.
Any ideas? Thanks.
"""
print(new_post)
new_post_vec = vectorizer.transform([new_post])
new_post_label = km.predict(new_post_vec)[0]
print("Label of new post is: %i" % new_post_label)

similar_indices = (km.labels_ == new_post_label).nonzero()[0]
similar = []
for i in similar_indices:
	dist = sp.linalg.norm((new_post_vec - vectorized[i]).toarray())
	similar.append((dist, train_data.data[i]))
similar = sorted(similar)
print("Count similar: %i" % len(similar))

show_at_1 = similar[0]
show_at_2 = similar[int(len(similar) / 10)]
show_at_3 = similar[int(len(similar) / 2)]

print("=== #1 ===")
print(show_at_1)
print()

print("=== #2 ===")
print(show_at_2)
print()

print("=== #3 ===")
print(show_at_3)

