import numpy as np
import pandas as pd
import itertools
from sklearn.datasets import load_iris
import plotly.express as px
import tqdm

class KMeans:
    def __init__(self, k):
        self.k = k
        self.means = None
        self.cluster_ids = None

    def get_cluster_ids(self):
        return self.cluster_ids

    def init_centroid(self,m):
        return np.random.randint(0,self.k,m)
    
    def cluster_means(self, X, clusters):
        m,n = X.shape[0], X.shape[1]
        temp = np.zeros((m,n+2))
        temp[:,:n], temp[:,n] = X, clusters
        result = np.zeros((self.k,n))
        for i in range(self.k):
            subset = temp[np.where(temp[:,-1]==i), :n]
            if subset[0].shape[0] > 0:
                result[i] = np.mean(subset[0],axis=0)
            else:
                result[i] = X[np.random.choice(X.shape[0],1,replace=True)]
        return result
    
    def compute_cluster(self, x):
        return min(range(self.k), key=lambda i: np.linalg.norm(x - self.means[i])**2)
    
    def fit(self, X):
        # pass
        m = X.shape[0]
        initial_clusters = self.init_centroid(m)
        new_clusters = np.zeros(initial_clusters.shape)
        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                self.means = self.cluster_means(X, initial_clusters)
                for i in range(m):
                    new_clusters[i] = self.compute_cluster(X[i])
                count_changed = (new_clusters != initial_clusters).sum()
                if count_changed == 0:
                    break
                initial_clusters = new_clusters
                t.set_description(f"changed: {count_changed} / {X.shape[0]}")
        self.cluster_ids = new_clusters

iris = load_iris(as_frame=True)
target = iris.target
target = target.apply(lambda s: "Setosa" if s == 0 else ("Versicolor" if s == 1 else "Virginica"))
X = iris.data.values
k = 3
model = KMeans(k=k)
model.fit(X)

cluster_ids = model.get_cluster_ids()
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = cluster_ids

fig = px.scatter(df, 
                 x='sepal length (cm)', 
                 y='sepal width (cm)',
                 color='Cluster',
                 symbol=target,
                 color_continuous_scale=px.colors.sequential.Viridis,
                 opacity=0.8)

fig.update_layout(xaxis_title="Sepal Length",
                  yaxis_title="Sepal Width",
                  coloraxis_showscale=False, 
                  title="Iris Plants (k = {})".format(k),
                  legend_title_text="Species")
fig.show()