import numpy as np


def dfs(adj_matrix, node, visited_nodes, subgraph_set):
    visited_nodes.add(node)
    subgraph_set.add(node)
    for i, connected in enumerate(adj_matrix[node]):
        if connected != 0 and i not in visited_nodes:
            dfs(adj_matrix, i, visited_nodes, subgraph_set)


def find_max_connected_subgraph(adj_matrix):
    num_nodes = len(adj_matrix)
    max_subgraph = set()
    visited = set()

    for i in range(num_nodes):
        if i not in visited:
            temp_subgraph = set()
            dfs(adj_matrix, i, visited, temp_subgraph)
            if len(temp_subgraph) > len(max_subgraph):
                max_subgraph = temp_subgraph

    return max_subgraph


if __name__ == '__main__':
    adj_matrix = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0]
    ])

    max_subgraph = find_max_connected_subgraph(adj_matrix)
    print(max_subgraph)
