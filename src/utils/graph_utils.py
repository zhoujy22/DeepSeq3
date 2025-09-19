import torch

def get_cone_list(x_data, ff_fanin_list, po_ff_list, pi_list):
    def dfs(node, visited, cone):
        if node in pi_list:
            return True, cone
        visited[node] = True
        for fanin_idx in ff_fanin_list[node]:
            if not visited[fanin_idx]:
                cone.insert(len(cone), fanin_idx)
                visited[fanin_idx] = True
                succ, cone = dfs(fanin_idx, visited, cone)
                if not succ:
                    cone.pop()
                    visited[fanin_idx] = False
        return True, cone
    
    cone_list = []
    for po_ff_idx in po_ff_list:
        cone = [po_ff_idx]
        visited = [False] * len(x_data)
        succ, cone = dfs(po_ff_idx, visited, cone)
        cone_list.append(cone)
        
    return cone_list

def detect_cycles(x_data, fanout_list,edge_data):

    def dfs(node, stack, visited, length, terminated_node):
        if node == terminated_node and length > 0:

            path = []
            for idx in range(len(stack)):
                if stack[idx]:
                    path.append(idx)
            path.sort()

            hash_value = get_hash(path)
            if hash_value not in cycle_hash_list:
                cycle_hash_list.append(hash_value)
                return 1, None, None,
            
            else:
                return 0, None,None
        
        for fanout_idx in fanout_list[node]:

            if not stack[fanout_idx] and not visited[fanout_idx]:
                stack[fanout_idx] = True
                visited[fanout_idx] = True

                res,_,_ = dfs(fanout_idx, stack, visited, length + 1, terminated_node)
                if res == 1:
                    return 1,  stack, visited
                elif res == 0:
                    stack[fanout_idx] = False
                    visited[fanout_idx] = False
                else:
                    stack[fanout_idx] = False
        return -1, None,None, 
    
    def get_hash(lst, mod=1e7+7):
        res = 0
        for item in lst:
            res = res * 23 + item * 31
            res %= int(mod)
        return res
                
    # detect
    no_cycle = 0
    cycle_hash_list = []

    
   
    edge_index = torch.tensor(edge_data, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    
    cyclic_FFs_nodes = [0] * len(x_data)
    #edges_in_cyclic_FFs = []
    edges_in_cycle = []
    for node in range(len(x_data)):
        visited = [False] * len(x_data)
        stack = [False] * len(x_data)
        res, stack, visited = dfs(node, stack, visited, 0, node) # this node is creating cycle if FF
        if res == 1:
            cyclic_FFs_nodes[node] = 1
            no_cycle += 1
           
            """
            path = []
            for idx in range(len(stack)):
                if stack[idx]:
                    path.append(idx)
            
            #print("FF nodes appearing in cycle: ",node,  path) ## all FFs appearing in cycle caused by node, multiple paths?

           
            # for p in path: # is it should be for all FFs appearing in a cycle or cycle originating FFs
            # currently for cycle originating FFs
            # getting all nodes in cyclic graph 
    
            visited = [False] * len(x_data)
            nodes_in_path = []

            result, nodes_in_path = find_nodes_in_cycle(node, node, edge_index, visited, nodes_in_path, flag= True)
            #print("nodes", nodes_in_path)
            if result:
                #getting all edges in cycle
                
                for edge in edge_data:
                    if int(edge[0]) in nodes_in_path and int(edge[1]) in nodes_in_path:
                        cyclic_FFs_nodes.append(node)
                        edges_in_cycle.append([edge[0],edge[1]])
                            
                #print(edges_in_cycle)
                #edges_in_cycle = torch.tensor(edges_in_cycle, dtype=torch.long)
                #edges_in_cycle = edges_in_cycle.t().contiguous()
                #edges_in_cyclic_FFs.append(edges_in_cycle)
            """    
        
    return no_cycle, cyclic_FFs_nodes
    #return no_cycle, cyclic_FFs_nodes, edges_in_cycle

def find_nodes_in_cycle(cyclic_ff, node, edge_index,visited, nodes_in_path, flag):
    if cyclic_ff == int(node) and not flag:
        return True, nodes_in_path
    
    src = edge_index[0]==node
    fanout_nodes = edge_index[1][src] #all fanouts of node

    for f in fanout_nodes:
        if not visited[f]:
            visited[f] = True
            res, nodes_in_path = find_nodes_in_cycle(cyclic_ff,f,edge_index,visited,nodes_in_path,flag=False)
            if res:
                nodes_in_path.append(int(f))
                return res, nodes_in_path
            else:
                not res, nodes_in_path
    return False, nodes_in_path

if __name__ == '__main__':
    x_data = [[0], [1], [2], [3], [4], [5], [6], [7]]
    fanout_list = [
        [1], [2], [3], [4, 1, 2], [2, 5], [6, 7], [7], [3]
    ]
    fanin_list = [
        [], [0, 3], [1, 3, 4], [2, 7], [3], [4], [5], [5, 6]
    ]
    pi_list = [0]
    po_ff_list = [6]
    cone_list = get_cone_list(x_data, fanin_list, po_ff_list, pi_list)
    print(cone_list)
    
    detect_cycles(x_data, fanout_list, None)

    