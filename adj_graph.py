from polygons import *
from aco_routing import ACO
import networkx as nx
import matplotlib.pyplot as plt
class Graph:
    def __init__(self):
        self.adjacency_list = {}

    def add_vertex(self, polygon):
        self.adjacency_list[polygon] = []

    def add_edge(self, polygon1, polygon2):
        self.adjacency_list[polygon1].append(polygon2)
        self.adjacency_list[polygon2].append(polygon1)

p1 = MyPolygon([[0,0],
               [0,100],
               [100,100],
               [100,0]])

p2 = MyPolygon([[100,50],
               [100,100],
               [150,100],
               [150,50]])

p3 = MyPolygon([[100,0],
               [100,50],
               [150,50],
               [150,0]])

p4 = MyPolygon([[150,50],
               [150,100],
               [200,100],
               [200,50]])

p5 = MyPolygon([[150,0],
               [150,50],
               [200,50],
               [200,0]])

p6 = MyPolygon([[200,0],
               [200,100],
               [300,100],
               [300,0]])

graph = Graph

polygons = {
  0: p1,
  1: p2,
  2: p3,
  3: p4,
  4: p5,
  5: p6,
}

G = nx.Graph()

G.add_edge("A", "B", cost=2)
G.add_edge("B", "C", cost=2)
G.add_edge("A", "H", cost=2)
G.add_edge("H", "G", cost=2)
G.add_edge("C", "F", cost=1)
G.add_edge("F", "G", cost=1)
G.add_edge("G", "F", cost=1)
G.add_edge("F", "C", cost=1)
G.add_edge("C", "D", cost=10)
G.add_edge("E", "D", cost=2)
G.add_edge("G", "E", cost=2)

# aco = ACO(G, ant_max_steps=100, num_iterations=100, ant_random_spawn=True)

# aco_path, aco_cost = aco.find_shortest_path(
#     source="A",
#     destination="D",
#     num_ants=100,
# )
# print(aco_path, aco_cost)

def create_block(parity):
    A = nx.Graph()
    A.add_node(1, pos = (0,0))
    A.add_node(2, pos = (0,2))
    A.add_node(3, pos = (1,2))
    A.add_node(4, pos = (1,0))

    A.add_edge(1,2, weight = 1)
    A.add_edge(3,4, weight = 1)
    L = 100
    e = 0
    if parity == 'even':
        e = 1
    if parity == 'odd':
        e = -1
    A.add_edge(2,3, weight = L**e)
    A.add_edge(1,4, weight = L**e)
    
    A.add_edge(1,3, weight = L**-e)
    A.add_edge(2,4, weight = L**-e)
    return A

G = create_block('even')

# G.add_edge("a", "b", weight=1)
# G.add_edge("a", "c", weight=2)
# G.add_edge("c", "d", weight=3)
# G.add_edge("c", "e", weight=4)
# G.add_edge("c", "f", weight=5)
# G.add_edge("a", "d", weight=6)

elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] > 0.5]
esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

pos=nx.get_node_attributes(G,'pos')

# nodes
nx.draw_networkx_nodes(G, pos, node_size=700)

# edges
nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6)
nx.draw_networkx_edges(
    G, pos, edgelist=esmall, width=6, alpha=0.5, edge_color="b", style="dashed"
)

# node labels
nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")
# edge weight labels
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(G, pos, edge_labels)

ax = plt.gca()
ax.margins(0.08)
plt.axis("off")
plt.tight_layout()
plt.show()