from torch_geometric.data import Data

class OrderedData(Data):

    def __init__(self, edge_index=None, x=None, y=None,  \
                 edge_attr=None, \
                 tt_pair_index=None, tt_diff=None, min_tt_dis=None, \
                 rc_pair_index=None, is_rc=None, \
                 finite_list=None, h_init=None, \
                 ff_pair_index = None,
                 is_ff_equ= None,
                 ff_sim= None,
                 forward_level=None, forward_index=None, backward_level=None, backward_index=None):
    
        
        super().__init__()

        self.edge_index = edge_index
        self.x = x
        self.y = y
        self.edge_attr = edge_attr
        self.tt_pair_index = tt_pair_index
        self.tt_diff = tt_diff
        self.min_tt_dis = min_tt_dis

        self.forward_level = forward_level
        self.forward_index = forward_index
        self.backward_level = backward_level
        self.backward_index = backward_index
        
        self.rc_pair_index = rc_pair_index
        self.is_rc = is_rc

        self.finite_list = finite_list
        self.h_init = h_init
        self.ff_pair_index = ff_pair_index
        self.is_ff_equ= is_ff_equ
        self.ff_sim= ff_sim
        
        #self.edges_in_cyclic_FFs = edges_in_cyclic_FFs,
    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        try:
            dim = None
            if key in ['forward_index', 'backward_index']:
                dim = 0
            elif key in ['edge_index', 'tt_pair_index', 'rc_pair_index', 'original_edge_index', 'diff_pair_index', 'ff_fanin_edge_index', 'ff_pair_index']:
                dim = 1
            else:
                dim = 0            
            return dim
        except Exception as e:
            print(f"[__cat_dim__ ERROR] key={key}, value={value}, error={e}")
            raise e
class OrderedData2(Data):

    def __init__(self, edge_index=None, x=None,  \
                 edge_attr=None, \
                 h_init1=None, h_init0=None,\
                 forward_level=None, forward_index=None, backward_level=None, backward_index=None):
    
        
        super().__init__()

        self.edge_index = edge_index
        self.x = x
        self.edge_attr = edge_attr
        

        self.forward_level = forward_level
        self.forward_index = forward_index
        self.backward_level = backward_level
        self.backward_index = backward_index
        
        self.h_init1 = h_init1
        self.h_init0 = h_init0
        
        #self.edges_in_cyclic_FFs = edges_in_cyclic_FFs,
    
    def __inc__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key:
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['edge_index']:
            return 1
        elif key in [
            'x', 'gate', 'y_prob', 'y_trans_prob', 'edge_attr', 'h_init',
            'forward_index', 'forward_level', 'backward_index', 'backward_level'
        ]:
            return 0
        else:
            return 0

