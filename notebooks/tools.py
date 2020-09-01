import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go



'''
definition:
    set width of notebooks
input: 
    width: int - width in percent
output: 
    nothing
'''
def set_screen_width(width=50):
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:" + width + "% !important; }</style>"))
    return



'''
definition:
    read dataset of events and split string with hashtags to array of hasgtags
input: 
    path: str - path to events dataset
output: 
    df: pandas.DataFrame - dataframe of events
'''
def read_events(path: str):
    df = pd.read_csv(path)
    df['tags'] = df['tags'].apply(lambda s: (s[1:-1]).replace("'", "").replace(' ', '').split(","))
    return df
    

    
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
    build dictionary: word -> semantic label
input: 
    sem_model_path: str - path to the sem_model_path model
output: 
    w2l: dict <str,int> - dictionary of semantic labels
'''
def build_word2label(sem_model_path:str):
    # read semantic labels for words
    with open(f'{sem_model_path}labels.txt') as f:
        labels = [line.split(',') for line in f]
    
    # filtering all exceptions (each row in labels.txt should look like 'word,label')
    labels = filter(lambda l: len(l) == 2, labels) 
    # cast words labels to int
    labels = list(map(lambda l: [l[0], int(l[1][:-1])], labels))
    
    # building dict: word -> label
    w2l = {}
    for word, label in labels:
        w2l[word] = label
        
    return w2l



'''
definition:
    calculate normalized (for using cosine distance) vectors events by semantic clusters of words
input: 
    df: pandas.DataFrame - dataframe of events
    sem_model_path: str - path to the semantic model
    n: int - number of semantic clusters
output: 
    df: pandas.DataFrame - dataframe of events wothut events for which not a single word is included in semantic clusters
    X: numpy 2d array - vectors for events
'''
def build_vectors_semantic(df: pd.DataFrame, sem_model_path:str, n:int):
    from sklearn.preprocessing import StandardScaler
    
    sem_path = f'{sem_model_path}{n}/'
    w2l = build_word2label(sem_path)
    
    # filtering words for which no semantic class is defined
    df['description'] = df['description'].apply(lambda s: ' '.join(filter(lambda w: w in w2l, s.split())))
    # removing empty events
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
    return df, X



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
    adds cluster labels to the event dataframe for each n of n_clusters by k-means algorithm on GPU
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
    adds cluster labels to the event dataframe for each n of n_clusters by hierarchical clustering on CPU
input: 
    n_clusters: list - a list of values of the number of clusters for which it is necessary 
        to cluster
    df: pandas.DataFrame - frame of events
    X: 2d numpy.array - vectors of events
    linkage: str - sklearn.cluster.AgglomerativeClustering.linkage
output: 
    df: pandas.DataFrame - frame of events with calculated cluster labels
'''
def agglomerative_list(n_clusters: list, df: pd.DataFrame, X: np.array, linkage='ward'):
    from sklearn.cluster import AgglomerativeClustering
    # return some element of set 
    def get_elem(s: set):
        return next(iter(s))
    
    # fitting model for n_clusters = 2
    model = AgglomerativeClustering(n_clusters=2, linkage=linkage)
    model = model.fit(X)
    
    n_samples = len(df)
    clusters = dict([(i, set([i])) for i in range(n_samples)])
    colors = list(range(n_samples))
    
    # restoring labels for all each elem of n_clusters by AgglomerativeClustering.children_ list
    n_clusters_set = set(n_clusters)
    for ind, pair in enumerate(model.children_):
        a, b = pair
        color_a, color_b = sorted([colors[get_elem(clusters[a])], colors[get_elem(clusters[b])]])
        clusters[ind + n_samples] = clusters[a] | clusters[b]
        colors = map(lambda c: c if c != color_b else color_a, colors)
        colors = list(map(lambda c: c - int(c > color_b), colors))
        if (n_samples - ind - 1) in n_clusters_set:
            df[str(n_samples - ind - 1)] = colors
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
        row = [n] + calc_scores(n, df[['id', str(n)]], pairs_df)
        rows.append(row)
    
    columns = ['n_clusters'] + data_scores_names
    df_scores = pd.DataFrame(rows, columns=columns)
    
    return df_scores



'''
definition:
    find n_clusters and value for given model and name of score
input: 
    scores_df: pandas.DataFrame - dataframe of events scores
    score_name: str - name of the score
output: 
    dict{'name': str, 'value': float, 'n_clusters': int}
'''
def find_best_score(scores_df: pd.DataFrame, score_name: str):
    best_score = scores_df.iloc[scores_df[score_name].idxmax()]
    best = { 'name': score_name, 'value': best_score[score_name], 'n_clusters': int(best_score['n_clusters'])}

    return best



'''
definition:
    find n_clusters, n_semantic, and value for given model and name of score
input: 
    n_semantic: list - list of semantic models
    tmp_path: str - path to tmp directory, where scores files were stored 
    df_name: str - name of the scores files
    score_name: str - name of score
output: 
    dict{'name': str, 'value': float, 'n_clusters': int, 'n_semantic': int}
'''
def find_best_semantic_score(n_semantic: list, tmp_path:str, df_name:str, score_name:str):
    best = {'name': score_name, 'value': 0, 'n_clusters': 0, 'n_semantic': 0}

    for n in n_semantic:
        suffix = f'{n}/'
        path = tmp_path + suffix
        scores_df = pd.read_csv(path + df_name)

        score = find_best_score(scores_df, score_name)
        if score[score_name] > best['value']:
            best = score
            best['n_semantic'] = n

    return best



'''
definition:
    create fig object for events clusters with function fig.show() for plotting
input: 
    df: pandas.DataFrame - dataframe of events with some labels and columns 'x' and 'y'
    color: int or str - name of df.columns with labels
    hover_name: str - name of df.columns with hover names
output: 
    plolty.express.figure
'''
def plot_events(df: pd.DataFrame, color:str or int, hover_name='title'):
    return px.scatter(df, x="x", y="y", color=str(color), hover_name=hover_name)



'''
definition:
    calculate cenrtoids for events clustering with most popular hashtags for given number of clusters
input: 
    df: pandas.DataFrame - dataframe of events with columns 'x' and 'y'
    n: int - the number of clusters
    hashtags_size: int - the count of the most popular hashtags to collect
output: 
    df_centroids: pandas.DataFrame - dataframe of centoids with their 2d coodinates, labes, size of events, and the most popular hashtags from events of this cluster
'''
def calc_centroids(df: pd.DataFrame, n:int, hashtags_size=15):
    from collections import Counter 
    df_centroids = pd.DataFrame([], columns=['x', 'y', 'label', 'hover_name', 'name', 'size', 'hashtags'])

    for i in range(n):
        cluster = df[df[str(n)] == i]
        x = cluster['x'].mean()
        y = cluster['y'].mean()
        ht = cluster['tags'].sum()
        
        name = Counter(ht).most_common(hashtags_size)
        hover_name = '<br>'.join(map(lambda p: p[0], name))
        df_centroids.loc[len(df_centroids)] = [x, y, float(i), hover_name, name, np.int64(len(cluster)), ht]

    df_centroids['size'] = df_centroids['size'].astype(np.int64)
    return df_centroids



'''
definition:
    create fig object for events centroids with function fig.show() for plotting
input: 
    df_centroids: pandas.DataFrame - dataframe of events centoids
    size_max: int - max size of centroids points on the figure
    size_text_tags: int - the number of hashtags written on centroids
    min_size: int - minimum numbers events in centroids, clusters with size < min_size are filtered. It is ignored if max_size is not equal None
    max_size: int or None: maximal numbers events in centroids, clusters with size > max_size are filtered.
output: 
    fig: plolty.express.figure
    df: pandas.DataFrame - dataframe of filtered centoids. If min_size and max_size have standard values, df is equal df_centroids
'''
def plot_centroids(df_centroids, size_max=100, size_text_tags=0, min_size=0, max_size=None):
    if max_size == None:
        df = df_centroids[df_centroids['size'] >= min_size]
    else:
        df = df_centroids[df_centroids['size'] <= max_size]
    df['text'] = df_centroids['hover_name'].apply(lambda s: '<br>'.join(s.split('<br>')[:size_text_tags]))
    fig = px.scatter(df, x="x", y="y", color='label', text='text', size='size', hover_name='hover_name', size_max=size_max)
    return fig, df



'''
definition:
    create fig object with line charts for events clustering scores with function fig.show() for plotting
input: 
    scores_df: pandas.DataFrame - dataframe of events clustering scores
    x: str - name of abscissa for scores_df
    y: str or list(str) - name or names scores for scores_df
    x_title: str - name of abscissa for plot 
    y_title: str - name of ordinate for plot
output: 
    plolty.express.figure
'''
def plot_score(scores_df: pd.DataFrame, y, x='n_clusters', x_title='n clusters', y_title='score value'):
    if type(y) == str:
        y = [y]
    fig = go.Figure() 
    for name in y:
        fig.add_trace(go.Scatter(x=scores_df[x], y=scores_df[name], name=name))
    
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig



'''
definition:
    create fig object with surface plot for events clustering scores of semantic model with function fig.show() for plotting
input: 
    n_semantic: list - 'y' of figure, different numbers of semantic clusters
    n_clusters: list - 'y' of figure, different numbers of events clusters
    tmp_path: str - path to tmp dir
    name_df: str - name of scores dataframe
    scores_name: str - name of score
output: 
    plolty.express.figure
'''
def plot_scores_3d(n_semantic: list, n_clusters: list, tmp_path: str, name_df: str, scores_name='f1', title='', xaxis_title='n clusters', yaxis_title='semantic clusters', zaxis_title='score'):
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