from PROJ10 import calculateQs, make_cluster
import numpy as np
import pandas as pd
import networkx as nx
import time

files = [(2, "KONKURS/D1-K=2.csv"), (7, "KONKURS/D2-K=7.csv"), (12, "KONKURS/D3-K=12.csv"), \
         (None, "KONKURS/D1-NLK.csv"), (None, "KONKURS/D2-NLK.csv"), (None, "KONKURS/D3-NLK.csv")]

for (com, ad_matrix) in files:
    print(ad_matrix)
    A=np.matrix(pd.read_csv(ad_matrix, header=None).values)
    G=nx.from_numpy_matrix(A)
    start = time.perf_counter()
    Qs  = calculateQs(G, communities=com)
    max_iter = max(Qs, key=lambda x: Qs[x][0])
    clusters = Qs[max_iter][1]
    clusters = make_cluster(G, clusters) + 1
    end = time.perf_counter()
    print(end-start, "sec")
    for (i,v) in enumerate(clusters):
        print(f"{i+1}, {v}")
