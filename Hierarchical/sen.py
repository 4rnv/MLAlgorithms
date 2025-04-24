import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
import pandas as pd

hierarchical = {'X': [79,60,4,20,70,11,30,5,1,1], 'Y': [4,32,71,1,15,6,14,88,1,1]}

df = pd.DataFrame(hierarchical)
print(df)

X=df.to_numpy()
plt.figure(figsize=(6,4))
plt.title("Final Dendogram")
dend = shc.dendrogram(shc.linkage(X, method='complete'))

X=df.to_numpy()
plt.figure(figsize=(6,4))
plt.title("Final Dendogram")
dend = shc.dendrogram(shc.linkage(X, method='average'))

X=df.to_numpy()
plt.figure(figsize=(6,4))
plt.title("Final Dendogram")
dend = shc.dendrogram(shc.linkage(X, method='single'))