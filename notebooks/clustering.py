import gensim.models
import gensim
from sklearn import cluster

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

DATA_PATH = '/data/'
MODEL_PATH = DATA_PATH + 'models/wv1/'
LEM_CAPTION_PATH = DATA_PATH + 'lem_caption/'
CAPTIONS = ['posts2016.csv', 'posts2017.csv', 'posts2018.csv', 'posts2019.csv', 'posts2020.csv']

model = gensim.models.Word2Vec.load(MODEL_PATH + 'mdl')
# X = model.wv[model.wv.vocab]
X = [] # positions in vector space
words = [] # keep track of words to label our data again later
for word in model.wv.vocab:
    X.append(model.wv[word])
    words.append(word)
    

print('start')
dbscan = cluster.DBSCAN(eps=0.5, min_samples=5, n_jobs=20, metric='cosine')
dbscan.fit(X)

data = list(zip(dbscan.labels_, words))

f = open(DATA_PATH + 'semantic_clusters.txt', 'w')
for label, word in data: 
    f.write(f'{word},{label}\n')

#import cudf
#from cuml.cluster import DBSCAN
#gdf_float = cudf.DataFrame()


print('success')