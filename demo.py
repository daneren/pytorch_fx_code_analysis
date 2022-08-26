import torch
# Simple module for demonstration
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.rand(3, 4))
        self.linear = torch.nn.Linear(4, 5)

    def forward(self, x):
        x = self.linear(x+self.param).clamp(min=0.0, max=1.0)
        
        return torch.nn.functional.relu(x)

module = MyModule()

from torch.fx import symbolic_trace
# Symbolic tracing frontend - captures the semantics of the module

import pdb; pdb.set_trace()
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)


# from torch.fx import Tracer
# tracer = Tracer()
# graph = tracer.trace(module)


# # High-level intermediate representation (IR) - Graph representation
# print(symbolic_traced.graph)

# transformer
#nodes = [node for node in symbolic_traced.graph.nodes]

#nodes[3].op = torch.div

#symbolic_trace.graph.lint()

#x = torch.fx.GraphModule(module, symbolic_trace.graph)

#print(x.graph)


import operator 
traced = symbolic_traced

patterns = set([operator.add, torch.add, "add"])

# Go through all the nodes in the Graph
for n in traced.graph.nodes:
    # If the target matches one of the patterns
    if any(n.target == pattern for pattern in patterns):
        # Set the insert point, add the new node, and replace all uses
        # of `n` with the new node
        with traced.graph.inserting_after(n):
            new_node = traced.graph.call_function(torch.bitwise_and, n.args, n.kwargs)
            n.replace_all_uses_with(new_node)
        # Remove the old node from the graph
        traced.graph.erase_node(n)

# Don't forget to recompile!
traced.recompile()

print(traced.graph)
