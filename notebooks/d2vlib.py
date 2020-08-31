import pandas as pd
import numpy as np



'''
definition:
    calculate normalized (for using cosine distance) vectors events by trained doc2vec model
input: 
    df: pd.DataFrame - dataframe of events
    d2v_path: str - path to the doc2vec model
output: 
    X: numpy 2d array - vectors for events
'''
def build_vectors_doc2vec(df: pd.DataFrame, d2v_path:str):
    from gensim.models.doc2vec import Doc2Vec
    
    model = Doc2Vec.load(d2v_path + 'mdl')
    
    X = [] # documents vectors
    for index, row in df.iterrows():
        words = row.description.split()
        vec = model.infer_vector(words)
        X.append(vec)  
    X = np.array(X)

    # vector normalization   
    X = X / (((X ** 2).sum(axis=1)) ** (1/2)).reshape(X.shape[0], 1)
    return X


'''
definition:
    calculate 2d embeddings for vectors of events for visualizations
input: 
    events_path: str - path to events dataset
    d2v_path: str - path to the doc2vec model
output: 
    df: pandas.DataFrame - frame for events
'''
def calculate_2d(df: pd.DataFrame, X: np.array):
    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=0, n_jobs=35, early_exaggeration=10, learning_rate=200)
    df['x'], df['y'] = list(zip(*tsne.fit_transform(X)))
    return df



'''
definition:
    adds cluster labels to the event dataframe for each n of n_clusters
input: 
    n_clusters: list - a list of values of the number of clusters for which it is necessary 
        to cluster
    df: pandas.DataFrame - frame of events
    X: 2d numpy.array - vectors of events
    n_init: int - number of centroid initializations for calculating kmeans
output: 
    df: pandas.DataFrame - frame of events with calculated cluster labels
'''
def k_means_list(n_clusters: list, df: pd.DataFrame, X: np.array, n_init=50):
    from cuml.cluster import KMeans
    
    for n in n_clusters:
        n = int(n)
        row = [n]
        labels = KMeans(n_clusters=n, n_init=50).fit(X).labels_
        df[str(n)] = labels
    
    return df



'''
definition:
    calculates metrics based on tagged event pairs for a given number of clusters
input: 
    n: int - number of clusters
    df: pandas.DataFrame - frame of events
    pair_df: pandas.DataFrame - frame of tagged pairs of events
output: 
    scores: list - return array of data_scores_names metrics 
'''
# names of metrics based on tagged event pairs
data_scores_names = ['precision', 'recall', 'f1', 'rand', 'tp', 'tn', 'fp', 'fn']
# defining tags for tagged event pairs
pair_labels = {'positive': 2, 'negative': 1}
def calc_scores(n: int, df: pd.DataFrame, pair_df: pd.DataFrame):
    events = df[['id', str(n)]].values.tolist()
    tp, tn, fp, fn = 0, 0, 0, 0
    d = dict(events)
    
    for _, row in pair_df.iterrows():
        a = row['id_a']
        b = row['id_b']
        l = row['label']
        
        tp += 1 if d[a] == d[b] and l == pair_labels['positive'] else 0
        tn += 1 if d[a] != d[b] and l == pair_labels['negative'] else 0
        fp += 1 if d[a] == d[b] and l == pair_labels['negative'] else 0
        fn += 1 if d[a] != d[b] and l == pair_labels['positive'] else 0
    
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)    
    rand = (tp + tn) / (tp + tn + fp + fn)
    return [precision, recall, f1, rand, tp, tn, fp, fn]



'''
definition:
    calculates metrics based on tagged event pairs for a given list of numbers of clusters
input: 
    n_clusters:list: int - list of numbers of clusters
    df: pandas.DataFrame - frame of events
    pair_df: pandas.DataFrame - frame of tagged pairs of events
output: 
    df_scores: pandas.DataFrame - return dataframe with data_scores_names for each number of clusters
'''
def calc_scores_list(n_clusters:list, df: pd.DataFrame, pairs_df: pd.DataFrame):
    rows = []
    for n in n_clusters:
        row = [n] + calc_scores(frame[['id', str(n)]], pairs_frame)
        rows.append(row)
    
    columns = ['n_clusters'] + data_scores_names
    df_scores = pd.DataFrame(rows, columns=columns)
    
    return df_scores