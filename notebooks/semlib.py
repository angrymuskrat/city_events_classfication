import tools
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.metrics import calinski_harabasz_score
from cuml.cluster import KMeans

from sklearn.cluster import AgglomerativeClustering


def prepare_model(tmp_path: str, events_path:str, sem_model_path:str, n: int):
    from sklearn.preprocessing import StandardScaler
    from sklearn.manifold import TSNE
    
    suffix = f'{n}/'
    path = tmp_path + suffix
    if not os.path.exists(path):
        os.makedirs(path)

    with open(f'{sem_model_path}{suffix}labels.txt') as f:
        labels = list(map(lambda l: [l[0], int(l[1][:-1])], filter(lambda l: len(l) == 2, [line.split(',') for line in f])))
    w2l = {} # word to label
    for word, label in labels:
        w2l[word] = label 


    # read events
    df = pd.read_csv(events_path)
    df['description'] = df['description'].apply(lambda s: ' '.join(filter(lambda w: w in w2l, s.split())))
    df = df[df['description'] != '']

    # calculate semantic vectors for events
    X = []
    for index, row in df.iterrows():
        vec = np.array([0] * n)
        words = list(filter(lambda w: w in w2l, row.description.split()))
        for word in words:
            vec[w2l[word]] += 1
        vec = vec / len(words)
        X.append(vec)
    X = np.array(X)

    # scaling of semantic vectors
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    

    # calculate 2d embeding for events
    tsne = TSNE(n_components=2, random_state=0, n_jobs=35, early_exaggeration=10, learning_rate=200)
    X_2d = tsne.fit_transform(X) 
    x_vals, y_vals = list(zip(*X_2d))
    df['x'] = x_vals
    df['y'] = y_vals
    return df, X


# calulate scores for n clusters of events
# return labels and list of values for calc_scrore_names
calc_scores_names = ['Calinski_Harabasz']
def k_means(n_clusters:int, X):
    kmeans = KMeans(n_clusters=n_clusters, n_init=50).fit(X)
    chs = calinski_harabasz_score(X, kmeans.labels_)
    return kmeans.labels_, [chs]


# calculate scores for list of different numbers of clusters - ns
# return pandas DataFrame, where row contain n_clustres and different scores for this n_clustres
def k_means_list(n_clusters: list, df, X):
    columns = ['n_clusters'] + calc_scores_names 
    ans_scores = []
    
    for n in n_clusters:
        l, scores = k_means(n, X)
        df[str(n)] = l
        ans_scores.append([n] + scores) 

    df_scores = pd.DataFrame(ans_scores, columns=columns) 
    return df, df_scores


def get_elem(s: set):
    return next(iter(s))


def agglomerative_list(n_clusters: list, df, X):
    needed_n_clusters = set(n_clusters)
    model = AgglomerativeClustering(n_clusters=2)
    model = model.fit(X)
    
    n_samples = len(df)
    clusters = dict([(i, set([i])) for i in range(n_samples)])
    colors = list(range(n_samples))
    
    for ind, pair in enumerate(model.children_):
        a, b = pair
        color_a, color_b = sorted([colors[get_elem(clusters[a])], colors[get_elem(clusters[b])]])
        clusters[ind + n_samples] = clusters[a] | clusters[b]
        colors = map(lambda c: c if c != color_b else color_a, colors)
        colors = list(map(lambda c: c - int(c > color_b), colors))
        if (n_samples - ind - 1) in needed_n_clusters:
            df[str(n_samples - ind - 1)] = colors
    return df


data_scores_names = ['precision', 'recall', 'f1', 'rand', 'tp', 'tn', 'fp', 'fn']
lname = {'positive': 2, 'negative': 1}
def calc_scores(events, pairs_frame):
    events = events.values.tolist()
    tp, tn, fp, fn = 0, 0, 0, 0
    d = {}
    for event, l in events:
        d[event] = l
    
    for _, row in pairs_frame.iterrows():
        a = row['id_a']
        b = row['id_b']
        l = row['label']
        
        if (not a in d) or (not b in d):
            continue
        
        tp += 1 if d[a] == d[b] and l == lname['positive'] else 0
        tn += 1 if d[a] != d[b] and l == lname['negative'] else 0
        fp += 1 if d[a] == d[b] and l == lname['negative'] else 0
        fn += 1 if d[a] != d[b] and l == lname['positive'] else 0
    
    #print(f'tp: {tp}\ntn: {tn}\nfp: {fp}\nfn: {fn}\n')
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)    
    rand = (tp + tn) / (tp + tn + fp + fn)
    #print(f'precision: {precision}\nrecall: {recall}\nf1: {f1}\n')
    return [precision, recall, f1, rand, tp, tn, fp, fn]
    
def calc_scores_list(n_clusters: list, frame, pairs_frame, scores=None):
    rows = []
    for n in n_clusters:
        row = calc_scores(frame[['id', str(n)]], pairs_frame)
        rows.append([n] + row)
    
    columns = ['n_clusters'] + data_scores_names
    transposed = list(zip(*rows))
        
    df_scores = pd.DataFrame(rows, columns=columns)
    
    if scores is not None: 
        for score_name in calc_scores_names:
            df_scores[score_name] = scores[score_name]
    
    return df_scores


def find_best_score(n_semantic: list, tmp_path:str, df_name:str, score_name:str):
    best = (0, 0, 0)

    for n in n_semantic:
        suffix = f'{n}/'
        path = tmp_path + suffix
        df_scores = pd.read_csv(path + df_name)

        best_score = df_scores.iloc[df_scores[score_name].idxmax()]
        best = (best_score[score_name], n, int(best_score['n_clusters'])) if best_score[score_name] > best[0] else best

    return best


def score_plot_3d(n_semantic: list, n_clusters: list, tmp_path: str, name_df: str, scores_name='f1', title='', xaxis_title='n clusters', yaxis_title='semantic clusters', zaxis_title='score'):
    x = np.array(n_clusters)
    y = np.array(n_semantic)
    z = []

    for n in y:
        suffix = f'{n}/'
        path = tmp_path + suffix
        df_scores = pd.read_csv(path + name_df)
        z.append(df_scores[scores_name].tolist())

    z = np.array(z)
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(
        title=title, 
        scene = dict(
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            zaxis_title=zaxis_title,
        ), height=700,)
    return fig