import torch

from .load import load_activation_prune_config, load_weight_prune_config
from .pruning_methods import weight_criteria_map, activation_criteria_map

from .sparse_parameterization import FakeSparseWeight, FakeStructuredSparseWeight


def prune_with_a_function(info, fn, sparsity):
    return fn(info, sparsity)
# fn: what pruning function
# sparsity: to what extent would it prunes


def get_weight_rank_fn(c):
    return weight_criteria_map[c["scope"]][c["granularity"]][c["method"]]
# import function: how to rank the weights' relative importance


def get_activation_rank_fn(c):
    return activation_criteria_map[c["scope"]][c["granularity"]][c["method"]]
# import function: how to rank activations' relative importance


def get_weight_hook(name, info, named_info, w_config: dict):
    # register parameterization
    w_rank_fn = get_weight_rank_fn(w_config)
    value = named_info["value"] # tensor
    w_sparsity = named_info["weight_sparsity"] # sparsity
    register_parameter_name = "weight"
    parameterization = FakeSparseWeight(w_rank_fn(value, info, w_sparsity)) # [tensor, info, sparsity]
    return (register_parameter_name, parameterization)


def get_weight_hook_channel(name, info, named_info, next_named_info, w_config: dict):
    # register parameterization
    w_rank_fn = get_weight_rank_fn(w_config)
    value = named_info["value"] # tensor
    if next_named_info != None:
        next_value = next_named_info["value"] # next_tensor
    else: # next_named_info=None
        next_value = None
    w_sparsity = named_info["weight_sparsity"] # sparsity
    register_parameter_name = "weight"
    parameterization = FakeSparseWeight(w_rank_fn(value, next_value, info, w_sparsity)) # [tensor, next_tensor, info, sparsity]
    return (register_parameter_name, parameterization)


def get_activation_hook(name, info, named_info, a_config: dict):
    a_rank_fn = get_activation_rank_fn(a_config)
    a_sparsity = named_info["activation_sparsity"]  # 始终是0.1

    value = named_info["value"]
    register_parameter_name = "register_forward_pre_hook"

    # a_rank_fn(value, info, a_sparsity)  the mask

    #parameterization = FakeSparseWeight(a_rank_fn(value, info, a_sparsity))
    #return (register_parameter_name, parameterization)
    
    # register forward hook
    def sparsify_input(module, args):
        if len(args) > 1:
            raise ValueError(
                f"{module.__class__.__name__} takes more than 1 argument at inference, the current sparsiy_input pre forward hook only allows one!"
            )
        #import pdb; pdb.set_trace()
        x = args[0]
        mask = a_rank_fn(x, info, a_sparsity)   # 问题所在
        module.activation_mask = mask
        # it seems like the output of this can be a non-tuple thing??
        return x * mask

    return ("register_forward_pre_hook", sparsify_input)
    


def build_pruning_hooks(info, w_config, a_config):
    # example of a_config: {'method': 'l1-norm', 'granularity': 'elementwise', 'scope': 'local', 'sparsity': 0.1}
    named_hooks = {}
    for k, v in info.items():
        #import pdb;pdb.set_trace()
        if v is not None:
            # for weights
            w_info = {
                "module_type": v["module_type"],
                "weight_sparsity": w_config["sparsity"],
                "value": v["weight_value"],
                "stats": v["weight_stats"],
                "shape": v["weight_shape"],
            }
            # for activations
            a_info = {
                "module_type": v["module_type"],
                "activation_sparsity": a_config["sparsity"],
                "value": v["activation_value"],
                "stats": v["activation_stats"],
                "shape": v["activation_shape"],
            }
            named_hooks[k] = {
                "w_hook": get_weight_hook(k, info, w_info, w_config),
                "a_hook": get_activation_hook(k, info, a_info, a_config),
            }
    return named_hooks


def build_pruning_hooks_kernel(info, w_config, a_config):
    # example of a_config: {'method': 'l1-norm', 'granularity': 'elementwise', 'scope': 'local', 'sparsity': 0.1}
    named_hooks = {}
    for k, v in info.items():
        if v is not None and v['module_type'] in ['conv2d']:
            # for weights
            w_info = {
                "module_type": v["module_type"],
                "weight_sparsity": w_config["sparsity"],
                "value": v["weight_value"],
                "stats": v["weight_stats"],
                "shape": v["weight_shape"],
            }
            # for activations
            a_info = {
                "module_type": v["module_type"],
                "activation_sparsity": a_config["sparsity"],
                "value": v["activation_value"],
                "stats": v["activation_stats"],
                "shape": v["activation_shape"],
            }
            named_hooks[k] = {
                "w_hook": get_weight_hook(k, info, w_info, w_config),
                "a_hook": get_activation_hook(k, info, a_info, a_config),
            }
    return named_hooks


def build_pruning_hooks_channel(info, w_config, a_config):
    named_hooks = {}
    tmp=list(info.items())

    for index, kvpair in enumerate(tmp):
        k = kvpair[0] ; v = kvpair[1]
        if v != None:
            if v['module_type'] in ['conv2d']:
                if index < len(tmp)-1:
                    for j in range(index + 1, len(tmp), 1):
                        #import pdb; pdb.set_trace()
                        if tmp[j][1]!=None and (tmp[j][1]['module_type'] in ['conv2d']):
                            next_kvpair = tmp[j]
                            next_k = next_kvpair[0]
                            next_v = next_kvpair[1]
                            break
                        if j==len(tmp)-1 and (not (tmp[j][1] is not None and tmp[j][1]['module_type'] in ['conv2d'])):
                            next_kvpair = None  
                else:
                    next_kvpair = None

                if next_kvpair != None:
                    # for weights
                    w_info = {
                        "module_type": v["module_type"],
                        "weight_sparsity": w_config["sparsity"],
                        "value": v["weight_value"],
                        "stats": v["weight_stats"],
                        "shape": v["weight_shape"],
                    }
                    next_w_info = {
                        "module_type": next_v["module_type"],
                        "weight_sparsity": w_config["sparsity"],
                        "value": next_v["weight_value"],
                        "stats": next_v["weight_stats"],
                        "shape": next_v["weight_shape"],
                    }

                    # for activations
                    a_info = {
                        "module_type": v["module_type"],
                        "activation_sparsity": a_config["sparsity"],
                        "value": v["activation_value"],
                        "stats": v["activation_stats"],
                        "shape": v["activation_shape"],
                    }
                    #import pdb; pdb.set_trace()
                    named_hooks[k] = {
                        "w_hook": get_weight_hook_channel(k, info, w_info, next_w_info, w_config),
                        "a_hook": get_activation_hook(k, info, a_info, a_config),
                    }

                else: # the last call_module
                    # for weights
                    w_info = {
                        "module_type": v["module_type"],
                        "weight_sparsity": w_config["sparsity"],
                        "value": v["weight_value"],
                        "stats": v["weight_stats"],
                        "shape": v["weight_shape"],
                    }

                    # for activations
                    a_info = {
                        "module_type": v["module_type"],
                        "activation_sparsity": a_config["sparsity"],
                        "value": v["activation_value"],
                        "stats": v["activation_stats"],
                        "shape": v["activation_shape"],
                    }

                    next_w_info=None
                    named_hooks[k] = {
                        "w_hook": get_weight_hook_channel(k, info, w_info, next_w_info, w_config),
                        "a_hook": get_activation_hook(k, info, a_info, a_config),
                    }

    return named_hooks


def fetch_info(node, module):
    # deal with conv2d
    if isinstance(module, torch.nn.Conv2d):
        a_value = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["value"]
        a_stats = node.meta["mase"].parameters["software"]["args"]["data_in_0"]["stat"]
        a_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]

        w_value = node.meta["mase"].parameters["common"]["args"]["weight"]["value"]
        w_stats = node.meta["mase"].parameters["software"]["args"]["weight"]["stat"]
        w_shape = node.meta["mase"].parameters["common"]["args"]["weight"]["shape"]
        return {
            "module_type": "conv2d",
            "weight_value": w_value,
            "weight_stats": w_stats,
            "weight_shape": w_shape,
            "activation_value": a_value,
            "activation_stats": a_stats,
            "activation_shape": a_shape,
        }

    # deal with linear
    if isinstance(module, torch.nn.Linear):
        a_value = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["value"]
        a_stats = node.meta["mase"].parameters["software"]["args"]["data_in_0"]["stat"]
        a_shape = node.meta["mase"].parameters["common"]["args"]["data_in_0"]["shape"]

        w_value = node.meta["mase"].parameters["common"]["args"]["weight"]["value"]
        w_stats = node.meta["mase"].parameters["software"]["args"]["weight"]["stat"]
        w_shape = node.meta["mase"].parameters["common"]["args"]["weight"]["shape"]
        return {
            "module_type": "linear",
            "weight_value": w_value,
            "weight_stats": w_stats,
            "weight_shape": w_shape,
            "activation_value": a_value,
            "activation_stats": a_stats,
            "activation_shape": a_shape,
        }

    # otherwise we just return None, and this module would be ignore in build_pruning_hooks
    return None


def prune_graph_iterator(graph, config: dict):
    # Setup all pruning-related parameters (incl. basic validation)
    w_config = load_weight_prune_config(config["weight"], graph)
    a_config = load_activation_prune_config(config["activation"], graph)

    # we need to loop twice, the first time is to fetch all necessary information
    # first loop
    info = {}
    for node in graph.fx_graph.nodes:
        # pruning only deals with modules at the moment
        if node.op == "call_module":
            module = graph.modules[node.target]
            meta = fetch_info(node, module)
            info[node.target] = meta
    
    #import pdb; pdb.set_trace()
    if w_config['granularity'] in ["channelwise"]:
        hooks = build_pruning_hooks_channel(info, w_config, a_config)
    elif w_config['granularity'] in ["kernelwise"]:
        hooks = build_pruning_hooks_kernel(info, w_config, a_config)
    else:
        # hook building
        hooks = build_pruning_hooks(info, w_config, a_config)
        # building_pruning_blocks确实是对于一切有weights和activations记录在案的module，都会执行prune；
        # 但是关键是记录的过程是由fetch_info完成的，fetch_info只对conv2d和linear做prune
        # (就是bulding_pruning_hooks是对info内的modules处理的，而info的记录过程在fetch_info内，fetch_info要求只对于conv2d和linear记录info)


    # prune in second loop by applying hooks to relevant modules
    #if w_config['method'] != 'channel_l1_weight':
    for node in graph.fx_graph.nodes:
        # pruning only deals with modules at the moment
        if node.op == "call_module":
            module = graph.modules[node.target]
            if isinstance(module, torch.nn.Conv2d): # 必须是conv2d
                name = node.target
                if name in hooks.keys():
                    node_hooks = hooks[name]
                    # check weight hook, if it exits, apply it
                    if node_hooks["w_hook"] is not None:
                        register_name, parameterization = node_hooks["w_hook"]
                        # apply weight pruning
                        torch.nn.utils.parametrize.register_parametrization(
                            graph.modules[node.target], register_name, parameterization
                        )
                    if node_hooks["a_hook"] is not None:
                        register_fn, hook_fn = node_hooks["a_hook"]
                        #register_name, parameterization = node_hooks["a_hook"]
                        # apply activation pruning
                        getattr(graph.modules[node.target], register_fn)(hook_fn)
                        #torch.nn.utils.parametrize.register_parametrization(
                        #    graph.modules[node.target], register_name, parameterization
                        #)
    '''
    else:
        for i, node in enumerate(graph.fx_graph.nodes):
            # pruning only deals with modules at the moment
            if node.op == "call_module":
                name = node.target
                for j in range(i + 1, len(graph.fx_graph.nodes, 1)):
                    next_node = graph.fx_graph.nodes[j]
                    if next_node.op == "call_module":
                        next_name = next_node.target
                        break
    '''

    return graph


def prune_transform_pass(graph, pass_args: dict = {}):
    """
    Apply pruning transformation to the given graph.
    This is achieved by adding a register_parametrization hook to weights
    and a register_pre_forward hook to activations

    :param graph: The input graph to be pruned.
    :type graph: MaseGraph

    :param pass_args: Optional arguments for the pruning transformation.
    :type pass_args: dict

    :return: The pruned graph and an empty dictionary.
    :rtype: tuple
    """
    graph = prune_graph_iterator(graph, pass_args)

    #print(type(graph.model))

    return graph, {}
