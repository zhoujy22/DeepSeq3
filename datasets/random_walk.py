import torch
from torch_geometric.utils import to_dense_adj
from torch_scatter import scatter

def get_random_walk_pe(x: torch.Tensor, edge_index: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    使用随机游走位置编码为每个节点生成 pe 向量。

    Args:
        x (torch.Tensor): 节点特征张量。
        edge_index (torch.Tensor): 图的边索引。
        k (int): 随机游走的最大步长，也是 pe 向量的维度。

    Returns:
        torch.Tensor: 每个节点的 pe 向量，形状为 [num_nodes, k]。
    """
    num_nodes = x.size(0)

    A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)
    deg = scatter(torch.ones_like(edge_index[0]), edge_index[0], dim_size=num_nodes)
    deg[deg == 0] = 1
    
    D_inv = torch.diag(1.0 / deg)
    
    P = torch.matmul(D_inv, A)

    pe_list = []
    P_k = P  

    for _ in range(k):
        pe_i = torch.diag(P_k)
        pe_list.append(pe_i)
        P_k = torch.matmul(P_k, P)

    pe = torch.stack(pe_list, dim=1)

    return pe

if __name__ == '__main__':
    x_example = torch.randn(4, 16)
    edge_index_example = torch.tensor([[0, 1, 1, 2, 2, 3],
                                   [1, 0, 2, 1, 3, 2]])
    pe_vector = get_random_walk_pe(x_example, edge_index_example, k=20)
    print(f"生成的 pe 向量形状: {pe_vector.shape}")
    print("每个节点的 pe 向量:\n", pe_vector)