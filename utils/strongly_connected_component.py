import numpy as np


def dfs1(adj_matrix, visited, node, stack):
    visited[node] = True

    for neighbor in range(len(adj_matrix)):
        if adj_matrix[node][neighbor] and not visited[neighbor]:
            dfs1(adj_matrix, visited, neighbor, stack)

    stack.append(node)


def dfs2(adj_matrix, visited, node, component):
    visited[node] = True
    component.append(node)

    for neighbor in range(len(adj_matrix)):
        if adj_matrix.T[node][neighbor] and not visited[neighbor]:
            dfs2(adj_matrix, visited, neighbor, component)


def get_largest_strongly_connected_component(adj_matrix):
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    stack = []

    # 第一次深度优先遍历
    for node in range(num_nodes):
        if not visited[node]:
            dfs1(adj_matrix, visited, node, stack)

    # 转置邻接矩阵
    adj_matrix_T = adj_matrix.T

    # 初始化访问标记数组
    visited = [False] * num_nodes

    largest_component = []

    # 第二次深度优先遍历
    while stack:
        node = stack.pop()
        if not visited[node]:
            current_component = []
            dfs2(adj_matrix_T, visited, node, current_component)

            if len(current_component) > len(largest_component):
                largest_component = current_component

    return largest_component


if __name__ == '__main__':
    # 示例邻接矩阵
    # adj_matrix = np.array([[0, 1, 1, 0],
    #                        [1, 0, 0, 1],
    #                        [0, 0, 0, 0],
    #                        [0, 0, 0, 0]])
    adj_matrix = np.array([[0, 1, 1, 1],
                           [1, 0, 1, 0],
                           [1, 1, 0, 0],
                           [1, 0, 0, 0]])

    largest_component = get_largest_strongly_connected_component(adj_matrix)
    print("最大强连通分量：", largest_component)
