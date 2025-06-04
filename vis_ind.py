'''
Example for visualizing an individual.
'''

import os
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize

path = os.path.join(os.path.dirname(__file__), 'examples/xor/results_xor_0')
# path = os.path.join(os.path.dirname(__file__), 'examples/cart2pole/results_cart_0')
gen = '99'

with open(f'{path}/data_ind_gen{gen:0>4}.pkl', 'rb') as pkl:
    ind = pickle.load(pkl)

for operon in ind.operonList:
    print(f'operon: {operon.id}')
#     for node in operon.nodeList:
#         print('node: {}'.format(node.id))
    for link in operon.linkList:
        print(f'link: {link.id} -- {link.fromNodeID} -> {link.toNodeID} ({link.weight})')

nodeList, linkList = [], []
for operon in ind.operonList:
    nodeList = np.concatenate([nodeList, operon.nodeList])
    linkList = np.concatenate([linkList, operon.linkList])

attributeOperonID = np.empty(len(nodeList))
for operon in ind.operonList:
    for node in operon.nodeList:
        attributeOperonID[node.id] = operon.id / (ind.maxOperonID + 1)

inputNodeList = [node for node in nodeList if node.type == 'input']
hiddenNodeList = [node for node in nodeList if node.type == 'hidden']
outputNodeList = [node for node in nodeList if node.type == 'output']


G = nx.DiGraph()

node_color = []
for i, node in enumerate(inputNodeList):
    G.add_node(node.id, pos=((i+0.5) * (10.0 / len(inputNodeList)), 0.0))
    node_color += [attributeOperonID[node.id]]

for i, node in enumerate(hiddenNodeList):
    G.add_node(node.id, pos=((i+0.5) * (10.0 / len(hiddenNodeList)) + 0.7, 0.7 + (i % 2) * 0.6))
    node_color += [attributeOperonID[node.id]]

for i, node in enumerate(outputNodeList):
    G.add_node(node.id, pos=((i+0.5) * (10.0 / len(outputNodeList)), 2.0))
    node_color += [attributeOperonID[node.id]]

# edge_colors = []
for i, link in enumerate(linkList):
    G.add_edge(link.fromNodeID, link.toNodeID, weight=link.weight)
    # edge_colors += [link.weight]


edge_pos = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] >= 0.0]
edge_neg = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] < 0.0]

weights_pos = [G[u][v]['weight'] for u, v in edge_pos]
weights_neg = [-G[u][v]['weight'] for u, v in edge_neg]

self_con = [(u, v) for (u, v, d) in G.edges(data=True) if u == v]
nodes_self_con = [G.nodes[u]['pos'] for u, v in self_con]
weights_self_con = [G[u][v]['weight'] for u, v in self_con]

# x, y = [], []
# for node in nodes_self_con:
#     x += [node[0]]
#     y += [node[1]]
#
# c_self_con = []
# for w in weights_self_con:
#     if w >= 0.0:
#         c_self_con += ['tab:blue']
#     else:
#         c_self_con += ['tab:orange']

x_pos, y_pos = [], []
x_neg, y_neg = [], []
w_pos, w_neg = [], []
for n, w in zip(nodes_self_con, weights_self_con):
    if w >= 0.0:
        x_pos += [n[0]]
        y_pos += [n[1]]
        w_pos += [w]
    else:
        x_neg += [n[0]]
        y_neg += [n[1]]
        w_neg += [w]

# plt.scatter(x, y, marker='o', c='w', s=600, linewidths=weights_self_con, edgecolors=c_self_con)

# plt.scatter(x_pos, y_pos, marker='o', c='w', s=600, linewidths=w_pos, edgecolors='tab:blue')
# plt.scatter(x_neg, y_neg, marker='8', c='w', s=600, linewidths=w_neg, edgecolors='tab:orange')

pos = nx.get_node_attributes(G, 'pos')
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_color, alpha=0.7, cmap=plt.cm.viridis)

# edges = nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10, edge_color=edge_colors, edge_cmap=plt.cm.coolwarm_r, edge_vmin=ind.minWeight, edge_vmax=ind.maxWeight)
# edges = nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10, width=edge_colors)
edges_pos = nx.draw_networkx_edges(G, pos, edgelist=edge_pos, arrowstyle='->',
                                   arrowsize=10, width=weights_pos, edge_color='tab:blue')
edges_neg = nx.draw_networkx_edges(G, pos, edgelist=edge_neg, arrowstyle='->',
                                   arrowsize=10, width=weights_neg, edge_color='tab:orange', style='--')

nx.draw_networkx_labels(G, pos, font_size=12)

# pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.coolwarm_r, norm=Normalize(vmin=ind.minWeight, vmax=ind.maxWeight))
# pc.set_array(edge_colors)
# plt.colorbar(pc)

if ind.maxOperonID == 0:
    plt.scatter([], [], c=[plt.cm.viridis(0.0)], label='Operon {}'.format(0), alpha=0.7)
else:
    for v in range(ind.maxOperonID + 1):
        plt.scatter([], [], c=[plt.cm.viridis(v/(ind.maxOperonID))], label='Operon {}'.format(v), alpha=0.7)

plt.legend(bbox_to_anchor=(-0.1, 1.1), loc='upper left', fontsize=12)

ax = plt.gca()
ax.set_axis_off()
plt.tight_layout()
plt.savefig(f'{path}/mbeann_ind_gen{gen:0>4}.pdf')
plt.show()
