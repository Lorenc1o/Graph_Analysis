import networkx as nx
import matplotlib.pyplot as plt
import torch
import time
from community import community_louvain
from torch_geometric.nn import GCNConv

def show_matrix_from_dict(matrix_dict, size, name):
    matrix = [[0 for i in range(size)] for j in range(size)]
    for i in range(size):
        for j in range(size):
            matrix[i][j] = matrix_dict[i][j]
    
    print(name)
    for row in matrix:
        print(row)

'''
PALLA algorithm
'''
def palla_algorithm(graph, k, verbose=False):
    # 1. Find all maximal cliques in G
    cliques = list(nx.find_cliques(graph))
    if verbose:
        print("Cliques:")
        print(cliques)

    # 2. Sort cliques by size
    cliques.sort(key=lambda clique: len(clique), reverse=True)
    cliques = [(i, clique) for i, clique in enumerate(cliques)]

    # 4. Create clique overlap matrix
    clique_overlap_matrix = {}
    for (i, clique) in cliques:
        clique_overlap_matrix[i] = {}
        for (j, other_clique) in cliques:
            if clique != other_clique:
                clique_overlap_matrix[i][j] = len(set(clique).intersection(set(other_clique)))
            else:
                clique_overlap_matrix[i][j] = len(clique)

    if verbose:
        show_matrix_from_dict(clique_overlap_matrix, len(cliques), "Clique overlap matrix")

    # 5. Compute clique connectivity matrix: C[i][j] = 1 if clique_overlap_matrix[i][j] >= k-1, 0 otherwise
    clique_connectivity_matrix = {}
    for (i, clique) in cliques:
        clique_connectivity_matrix[i] = {}
        for (j, other_clique) in cliques:
            if i == j and clique_overlap_matrix[i][j] >= k:
                clique_connectivity_matrix[i][j] = 1
            elif i != j and clique_overlap_matrix[i][j] >= k-1:
                clique_connectivity_matrix[i][j] = 1
            else:
                clique_connectivity_matrix[i][j] = 0

    if verbose:
        show_matrix_from_dict(clique_connectivity_matrix, len(cliques), "Clique connectivity matrix")

    # 7. Compute communities: merge cliques that are connected
    clique_communities = {i: set(clique) for (i, clique) in cliques}
    for (i, clique) in cliques:
        for (j, other_clique) in cliques:
            if clique_connectivity_matrix[i][j] == 1:
                clique_communities[i] = clique_communities[i].union(clique_communities[j])
                clique_communities[j] = clique_communities[i]

    # 8. Remove duplicate communities
    clique_communities = list(set([tuple(community) for community in clique_communities.values()]))
    communities = []

    # 9. Return communities as list of sets
    for community in clique_communities:
        if len(set(community)) >= k:
            communities.append(set(community))

    return communities


'''
PALLA algorithm using PyTorch

This implementation uses PyTorch to compute the clique overlap matrix, clique connectivity matrix, and reachability matrix.
The algorithm is the same as the original PALLA algorithm, but the matrix operations are performed using PyTorch tensors.

Much more efficient than the original implementation, especially for large graphs.
'''
def palla_algorithm_pytorch(graph, k, verbose=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Find all maximal cliques in G (this part remains unchanged as it's graph-specific logic)
    cliques = list(nx.find_cliques(graph))
    num_cliques = len(cliques)
    if verbose:
        print("Cliques:")
        print(cliques)

    # Prepare a tensor for the clique overlap matrix
    clique_overlap_matrix = torch.zeros((num_cliques, num_cliques), device=device)

    # Populate the clique overlap matrix
    for i in range(num_cliques):
        for j in range(i, num_cliques):  # No need to compute when j < i due to symmetry
            if i != j:
                overlap = len(set(cliques[i]).intersection(set(cliques[j])))
                clique_overlap_matrix[i, j] = overlap
                clique_overlap_matrix[j, i] = overlap  # Symmetric assignment
            else:
                clique_overlap_matrix[i, j] = len(cliques[i])

    if verbose:
        print("Clique overlap matrix:\n", clique_overlap_matrix.cpu().numpy())

    # Compute clique connectivity matrix
    connectivity_condition = (clique_overlap_matrix >= (k-1)).float()
    # For diagonal elements where the overlap is the size of the clique, adjust for k
    for i in range(num_cliques):
        if clique_overlap_matrix[i, i] < k:
            connectivity_condition[i, i] = 0

    if verbose:
        print("Clique connectivity matrix:\n", connectivity_condition.cpu().numpy())

    # We start with the reachibility matrix as the diagonal of the connectivity matrix
    reachability_matrix = torch.diag(connectivity_condition.diagonal())
    
    # Incremental power matrix to add new paths
    incremental_power_matrix = connectivity_condition.clone()

    for _ in range(num_cliques - 1):  # In the worst case, the diameter is num_cliques - 1
        reachability_matrix += incremental_power_matrix
        # Compute the next power
        incremental_power_matrix = torch.matmul(incremental_power_matrix, connectivity_condition)

    # Threshold the reachability matrix to binary values: connected or not
    reachability_matrix = (reachability_matrix > 0).float()

    # Now, reachability_matrix indicates connected components (communities)
    # Extract communities from the reachability matrix
    communities = []
    visited = torch.zeros(num_cliques, dtype=torch.bool, device=device)

    for i in range(num_cliques):
        if not visited[i]:
            # Find all cliques connected to i
            connected_cliques = (reachability_matrix[i] > 0).nonzero(as_tuple=True)[0]
            community = set()
            for idx in connected_cliques:
                visited[idx] = True
                community = community.union(cliques[idx.item()])
            communities.append(community)
    # Delete communities with less than k nodes
    communities = [community for community in communities if len(community) >= k]
    return communities

def show_graph(graph):
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()

def show_communities(graph, communities):
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
    color_map = {}
    for i, community in enumerate(communities):
        for node in community:
            color_map[node] = colors[i]
    
    nx.draw(graph, node_color=[color_map[node] for node in graph.nodes()], with_labels=True, font_weight='bold')
    plt.show()

def show_communities_separate(graph, communities):
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
    # Prepare the base color map as light gray for all nodes
    base_color_map = {node: 'lightgray' for node in graph.nodes()}

    for i, community in enumerate(communities):
        # Copy the base color map to ensure all non-community nodes are light gray
        color_map = base_color_map.copy()
        # Update the color map for nodes in the current community
        for node in community:
            color_map[node] = colors[i % len(colors)]
        
        # Draw the graph for the current community
        plt.figure(figsize=(8, 6))
        nx.draw(graph, node_color=[color_map[node] for node in graph.nodes()], with_labels=True, font_weight='bold')
        plt.title(f'Community {i+1}')
        plt.show()

def show_communities_side_by_side(graph, communities):
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan']
    base_color_map = {node: 'lightgray' for node in graph.nodes()}

    # Compute node positions using a layout algorithm with a fixed seed for consistency
    pos = nx.spring_layout(graph, seed=42)  # You can change the layout algorithm if needed

    num_communities = len(communities)
    fig, axes = plt.subplots(1, num_communities, figsize=(num_communities * 5, 5))
    
    if num_communities == 1:  # If there's only one community, axes is not a list
        axes = [axes]
    
    for i, (community, ax) in enumerate(zip(communities, axes)):
        color_map = base_color_map.copy()
        for node in community:
            color_map[node] = colors[i % len(colors)]
        
        # Draw the graph on the subplot using the fixed positions
        nx.draw(graph, ax=ax, pos=pos, node_color=[color_map[node] for node in graph.nodes()], with_labels=True, font_weight='bold')
        ax.set_title(f'Community {i+1}')
    
    plt.tight_layout()
    plt.show()

# Graph AutoEncoder
class GAE(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GAE, self).__init__()
        self.encoder = GCNConv(in_channels, out_channels, cached=True)
        self.decoder = torch.nn.Linear(out_channels, in_channels, bias=False)
    
    def forward(self, x, edge_index):
        z = self.encoder(x, edge_index)
        z = torch.relu(z)
        adj_logits = self.decoder(z)
        adj_logits = torch.softmax(adj_logits, dim=1)
        return adj_logits

if __name__ == '__main__':
    #graph = nx.Graph()
    # Example 1: 1 community of size 4
    #graph.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (3, 4)])
    #print(palla_algorithm(graph, 3))
    #show_graph(graph)
    #show_communities(graph, palla_algorithm(graph, 3))

    # Example 2: 2 communities of size 4
    #graph.add_edges_from([(4, 5),(5, 6), (5, 7), (5, 8), (6, 7), (7, 8)])
    #print(palla_algorithm(graph, 3))
    #show_graph(graph)
    #show_communities(graph, palla_algorithm(graph, 3))

    # Example 3: from the slides
    #graph2 = nx.Graph()
    #graph2.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (2, 6), (3, 4), (3, 5), (3, 6), (4, 5), (5, 6), (4, 6), (6, 7), (6, 8), (6, 9), (6, 10), (7, 8), (7, 9), (8, 9), (8, 10), (9, 10)])
    #show_graph(graph2)
    #print(palla_algorithm(graph2, 4))
    #show_communities(graph2, palla_algorithm(graph2, 3, verbose=True))

    # Example 4: from the slides
    graph3 = nx.Graph()
    graph3.add_edges_from([(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (2, 9), (2, 10), (3, 4), (3, 5), (3, 6), (4, 5), (4, 6), (4, 7), (4, 9), (4, 8), (5, 6), (5, 7), (5, 9), (6, 7), (6, 9), (7, 9), (7, 8), (8, 9), (9, 10)])
    show_graph(graph3)
    time0 = time.time()
    communities = palla_algorithm(graph3, 4, verbose=True)
    print("Time:", time.time() - time0)
    print(communities)
    show_communities_side_by_side(graph3, communities)
    time0 = time.time()
    communities_torch = palla_algorithm_pytorch(graph3, 4, verbose=True)
    print("Time:", time.time() - time0)
    print(communities_torch)
    show_communities_side_by_side(graph3, communities_torch)

    # Real example: ego-Facebook dataset from SNAP
    # Read ../data/facebook/0.edges, 0.circles, 0.egofeat, 0.feat, 0.featnames
    graph4 = nx.read_edgelist('data/facebook/0.edges')
    time0 = time.time()
    communities = palla_algorithm_pytorch(graph4, 4, verbose=True)
    print("Time:", time.time() - time0)

    # Louvain algorithm
    time0 = time.time()
    partition = community_louvain.best_partition(graph4)
    print("Time:", time.time() - time0)

    modularity_palla = community_louvain.modularity({i: list(community) for i, community in enumerate(communities)}, graph4)
    print("Modularity Palla:", modularity_palla)
    print("Modularity Louvain:", community_louvain.modularity(partition, graph4))