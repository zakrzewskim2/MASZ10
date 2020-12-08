#%%
import numpy as np
import networkx as nx
import random

def gen_communityER(N, p_inside, p_outside, communities):
    community_size = N / communities

    G = nx.Graph()

    for i in range(N):
        G.add_node(i)

    for i in range(N):
        for j in range(i + 1, N):
            if i // community_size == j // community_size:
                if random.random() < p_inside:
                    G.add_edge(i, j)
            else:
                if random.random() < p_outside:
                    G.add_edge(i, j)
    
    clusters = np.repeat(np.arange(0, communities), community_size)
    return G, clusters

communities = 5
G, culsters = gen_communityER(100, 0.8, 0.01, communities)
nx.draw(G)
# %%
def calculatE(G, culsters, communities):
    num_of_edges = G.number_of_edges()
    adjacency_matrix = nx.adjacency_matrix(G) #np.array(nx.adjacency_matrix(G).todense())
    e = []
    indices = [list(culsters[i]) for i in range(communities)]
    for i in range(communities):
        e.append([0]*i)
        cl1_indices = indices[i]
        for j in range(i, communities):
            cl2_indices = indices[j]
            edges_between_communities = np.sum(adjacency_matrix[cl1_indices,:][:,cl2_indices])
            e[-1].append(edges_between_communities/num_of_edges)
    return np.array(e)

# %%
import pandas as pd
import time
def calculateQ(G, connected_components, communities):
    e = calculatE(G, connected_components, communities)
    return np.trace(e) - np.sum(np.power(e, 2))

def calculateQs(G_orig, communities=None, batch_size=1, cut_treshold=0.25):
    G = G_orig.copy()
    Qs = {}
    i = 0
    connected_components = [*nx.connected_components(G)]

    while G.number_of_edges()/G_orig.number_of_edges() > cut_treshold:
        if i%10==0:
            print(G.number_of_edges())
        betweeness = nx.edge_betweenness_centrality(G)
        max_betweeness_edge = sorted(betweeness, key = lambda x: betweeness[x], reverse=True)
        components_before = len(connected_components)
        if type(batch_size) == int:
            G.remove_edges_from(max_betweeness_edge[0:batch_size])
        else:
            G.remove_edges_from(max_betweeness_edge[0:max(int(batch_size*G.number_of_edges()),1)])
        connected_components = [*nx.connected_components(G)]
        components_after = len(connected_components)
        if components_before < components_after and communities is None:
            q = calculateQ(G_orig, connected_components, components_after)
            Qs[i] = (q,connected_components)
        elif communities is not None and components_after >= communities:
            q = calculateQ(G_orig, connected_components, components_after)
            Qs[i] = (q,connected_components)
            return Qs
        i += 1
    return Qs
#%%
# #ASOIAF
# nodes_book_1 = pd.read_csv("asoiaf-book1-nodes.csv")
# edges_book_1 = pd.read_csv("asoiaf-book1-edges.csv")
# string_nodes = {node[1].Id:i for i,node in enumerate(nodes_book_1.iterrows())}
# G.add_nodes_from(string_nodes.values())
# G.add_edges_from([(string_nodes[edge[1].Source], string_nodes[edge[1].Target]) for edge in edges_book_1.iterrows()])
# name = "ASOIAF"

# #EMAILS
# G = nx.parse_edgelist(open("email-Eu-core.txt").readlines(), nodetype=int)
# name = "EMAILS"

# #FB-PAGES
# G = nx.parse_edgelist(open("fb-pages-food.csv").readlines(), nodetype=int)
# name = "FB"

# #DOLPHINS
# G = nx.parse_edgelist(open("soc-dolphins.mtx").readlines(), nodetype=int)
# name = "DOLPHINS"

# #WIKI
# G = nx.parse_edgelist(open("soc-wiki-Vote.mtx").readlines(), nodetype=int)
# name = "WIKI"

#FOOTBALL
G_raw = nx.read_gml("football.gml")
G = nx.Graph()
string_nodes = {node:i for i,node in enumerate(G_raw.nodes())}
G.add_nodes_from(string_nodes.values())
G.add_edges_from([(string_nodes[edge[0]], string_nodes[edge[1]]) for edge in G_raw.edges()])
name = "FOOTBALL"


# #LES MISERABLES
# G_raw = nx.read_gml("lesmis.gml")
# G = nx.Graph()
# string_nodes = {node:i for i,node in enumerate(G_raw.nodes())}
# G.add_nodes_from(string_nodes.values())
# G.add_edges_from([(string_nodes[edge[0]], string_nodes[edge[1]]) for edge in G_raw.edges()])
# name = "LES MISERABLES"


# #ARTIFICIAL COMMUNITIES
# G,_= gen_communityER(50, 0.7, 0.05, 5)
# name = "ARTIFICIAL"


start = time.perf_counter()
Qs  = calculateQs(G, batch_size=0.1)
end = time.perf_counter()
print(end-start, "sec")
max_iter = max(Qs, key=lambda x: Qs[x][0])
print("Q", Qs[max_iter][0],"number of classes", len(np.unique(Qs[max_iter][1])))
# %%
def make_cluster(components_after):
    clusters = [None] * G.number_of_nodes()
    for (i,c) in enumerate(components_after):
        for v in c:
            clusters[v] = i
    return np.array(clusters)

def get_nice_layout(G, cluster):
    layouts = []
    for c in cluster:
        sub_g = nx.subgraph(G, c)
        layouts.append(nx.spring_layout(sub_g))
    complete_layout = nx.spring_layout(nx.complete_graph(len(cluster)))
    final_layout = {}
    for i,center in enumerate(complete_layout.values()):
        for (k,v) in layouts[i].items():
            final_layout[k] = (center[0]+v[0]/np.sqrt(len(cluster)), center[1]+v[1]/np.sqrt(len(cluster)))
    return final_layout


import matplotlib.pyplot as plt
plt.figure(figsize=(40,40))
clusters = Qs[max_iter][1]
nx.draw(G, node_color=make_cluster(clusters), pos=get_nice_layout(G, clusters))
plt.savefig(f"{name}.png")
# %%
