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


def create_graph(lines):
    A = nx.Graph()
    cnt = 0
    for i in range(len(lines)):  
        A.add_node((i,0), pos = (i,0))
        A.add_node((i,1), pos = (i,1))
    for i in range(len(lines)):
        A.add_edge((i,0), (i,1), cost = 1)
        for j in range(i+1, len(lines)):  
            A.add_edge((i,0), (j,0), cost = round(np.hypot(lines[i].Point1.x-lines[j].Point1.x, lines[i].Point1.y-lines[j].Point1.y), 1))
            A.add_edge((i,0), (j,1), cost = round(np.hypot(lines[i].Point1.x-lines[j].Point2.x, lines[i].Point1.y-lines[j].Point2.y), 1))
            A.add_edge((i,1), (j,0), cost = round(np.hypot(lines[i].Point2.x-lines[j].Point1.x, lines[i].Point2.y-lines[j].Point1.y), 1))
            A.add_edge((i,1), (j,1), cost = round(np.hypot(lines[i].Point2.x-lines[j].Point2.x, lines[i].Point2.y-lines[j].Point2.y), 1))
    return A


def plot_graph(G):
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d["cost"] > 0.5]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["cost"] <= 0.5]

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
    edge_labels = nx.get_edge_attributes(G, "cost")
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

def create_block(parity):
    A = nx.Graph()
    A.add_node(1, pos = (0,0))
    A.add_node(2, pos = (0,2))
    A.add_node(3, pos = (1,2))
    A.add_node(4, pos = (1,0))

    A.add_edge(1,2, cost = 1)
    A.add_edge(3,4, cost = 1)
    L = 100
    e = 0
    if parity == 'even':
        e = 1
    if parity == 'odd':
        e = -1
    A.add_edge(2,3, cost = L**e)
    A.add_edge(1,4, cost = L**e)
    
    A.add_edge(1,3, cost = L**-e)
    A.add_edge(2,4, cost = L**-e)
    return A

G = create_block('even')

aco = ACO(G, ant_max_steps=100, num_iterations=100, ant_random_spawn=True)

aco_path, aco_cost = aco.find_shortest_path(
    source=1,
    destination=3,
    num_ants=100,
)

# plot_graph(G)

