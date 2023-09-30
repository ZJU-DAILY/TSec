from typing import List

import numpy as np


def get_largest_complete_subgraph(adj_matrix: np.ndarray) -> List[int]:
    num_nodes = len(adj_matrix)
    complement_matrix = 1 - adj_matrix  # 补图的邻接矩阵

    max_clique_size = 0
    max_clique = []

    # 遍历所有节点
    for node in range(num_nodes):
        clique = [node]  # 当前团的节点列表

        # 对于当前节点，找到与其相邻的节点
        for neighbor in range(num_nodes):
            if complement_matrix[node][neighbor] == 0:
                clique.append(neighbor)

        # 判断当前团是否是一个完全子图
        is_complete = True
        for i in range(len(clique)):
            for j in range(i + 1, len(clique)):
                if adj_matrix[clique[i]][clique[j]] == 0:
                    is_complete = False
                    break
            if not is_complete:
                break

        # 如果当前团是完全子图，并且比已知的最大完全子图更大，则更新最大完全子图
        if is_complete and len(clique) > max_clique_size:
            max_clique_size = len(clique)
            max_clique = clique

    return max_clique


if __name__ == '__main__':
    # 示例邻接矩阵
    # adj_matrix = np.array([[0, 1, 1, 0],
    #                        [1, 0, 1, 0],
    #                        [1, 1, 0, 1],
    #                        [0, 0, 1, 0]])
    adj_matrix = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 0],
                           [1, 1, 0, 0],
                           [1, 0, 0, 0]])

    largest_complete_subgraph = sorted(get_largest_complete_subgraph(adj_matrix))
    print("最大完全子图：", largest_complete_subgraph)
