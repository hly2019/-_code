import networkx as nx
import numpy as np
G = nx.Graph()
# c = nx.Graph()
G.add_edge((1, 2), "a", capacity=np.inf)
# G.add_edge("a", (1, 2), capacity=3.0)
G.add_edge((1, 2), "b", capacity=np.inf)
G.add_edge((1, 2), "a", capacity=np.inf)
G.add_edge("a", "c", capacity=3.0)
G.add_edge("b", "c", capacity=5.0)
G.add_edge("b", "d", capacity=4.0)
G.add_edge("d", "e", capacity=2.0)
G.add_edge("c", "y", capacity=np.inf)
G.add_edge("e", "y", capacity=np.inf)
cut_value, partition = nx.minimum_cut(G, (1, 2), "y")
reachable, non_reachable = partition
print(reachable, cut_value)
a = "x" in G
print(a)
for node in G:
    print(node)