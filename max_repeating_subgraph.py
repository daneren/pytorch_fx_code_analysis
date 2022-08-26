import torch
import torchvision

from torch.fx import symbolic_trace


import queue

class BinaryTree:

    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None


    # def from_list_create_tree(self, node_list):
        
    #     root = BinaryTree(node)

    #     for next_node in list(node.users.keys())[0]:
    #         if next_node.all_input_nodes[0] == node:    
    #             if root.left_child == None:
    #                 root.insert_left(next_node)
    #             else:
    #                 root.insert_right(next_node)


    def from_graph_create_tree(self, graph):

        node_list = [node for node in graph.nodes]

        self.value = BinaryTree(node_list[0])

        for node in node_list:
            for next_node in list(node.users.keys())[0]:
                if next_node.all_input_nodes[0] == node:    
                    if self.left_child == None:
                        self.left_child = next_node
                    else:
                        self.right_child = next_node
            



    def insert_left(self, value):
        if self.left_child == None:
            self.left_child = BinaryTree(value)
        else:
            new_node = BinaryTree(value)
            new_node.left_child = self.left_child
            self.left_child = new_node

    
    def insert_right(self, value):
        if self.right_child == None:
            self.right_child = BinaryTree(value)
        else:
            new_node = BinaryTree(value)
            new_node.right_child = self.right_child
            self.right_child = new_node

    def bfs(self):
        queue = queue.Queue()
        queue.put(self)

        while not queue.empty():
            current_node = queue.get()
            print(current_node.value)

            if current_node.left_child:
                queue.put(current_node.left_child)

            if current_node.right_child:
                queue.put(current_node.right_child)


module = torchvision.models.resnet50()


import pdb;pdb.set_trace()
# x = torch.randn(1, 3, 224, 224, requires_grad=True)
# script_module = torch.jit.trace(module,x)
# torch.jit.save(script_module, "model.pt")
# print("torch.jit.save success") 


# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced : torch.fx.GraphModule = symbolic_trace(module)

# High-level intermediate representation (IR) - Graph representation
# print(symbolic_traced.graph)

symbolic_traced_modules = dict(symbolic_traced.named_modules())

for i, node in enumerate(symbolic_traced.graph.nodes):
    if node.op == 'placeholder':
        node.kind = "placeholder"
    elif node.op == 'output':
        node.kind = "output"
    elif node.op == 'call_function':
        node.kind = node.target
    elif node.op == 'call_module':
        node.kind = type(symbolic_traced_modules[node.target])
    elif node.op == 'call_method':
        node.kind = node.target
    elif node.op == 'get_attr':
        raise ValueError(f"{node} is get_attr")
    else:
        raise ValueError(f"{node} op is unknown")

    print(node.name)

node_list = [node for node in symbolic_traced.graph.nodes]

node_type_list = [node.kind for node in symbolic_traced.graph.nodes]

print(node_type_list)

type_dict = {}

node_type_str = ""

for node_type in node_type_list:
    if node_type in type_dict.keys():
        node_type_str = node_type_str + type_dict[node_type]
    else:
        type_dict[node_type] = chr(len(type_dict) + 97)
        node_type_str = node_type_str + type_dict[node_type]

print(node_type_str)
print(type_dict)
# for i, node in enumerate(node_list):
#     print(f"{i} {node.kind}")

# node_list.reverse()

# for node in node_list:

#     root = BinaryTree(node)

#     for next_node in list(node.users.keys())[0]:
#         if next_node.all_input_nodes[0] == node:
#             if root.left_child == None:
#                 root.insert_left(next_node)
#             else:
#                 root.insert_right(next_node)

# for node in node_list:

#     for next_node in list(node.users.keys())[0]:
#         if next_node.all_input_nodes[0] == node:
import random
class Solution:
    def longestDupSubstring(self, s: str) -> str:
        # 生成两个进制
        a1, a2 = random.randint(26, 100), random.randint(26, 100)
        # 生成两个模
        mod1, mod2 = random.randint(10**9+7, 2**31-1), random.randint(10**9+7, 2**31-1)
        n = len(s)
        # 先对所有字符进行编码
        arr = [ord(c)-ord('a') for c in s]
        # 二分查找的范围是[1, n-1]
        l, r = 1, n-1
        length, start = 0, -1
        while l <= r:
            m = l + (r - l + 1) // 2
            idx = self.check(arr, m, a1, a2, mod1, mod2)
            # 有重复子串，移动左边界
            if idx != -1:
                l = m + 1
                length = m
                start = idx
            # 无重复子串，移动右边界
            else:
                r = m - 1
        return s[start:start+length] if start != -1 else ""

    def check(self, arr, m, a1, a2, mod1, mod2):
        n = len(arr)
        aL1, aL2 = pow(a1, m, mod1), pow(a2, m, mod2)
        h1, h2 = 0, 0
        for i in range(m):
            h1 = (h1 * a1 + arr[i]) % mod1
            h2 = (h2 * a2 + arr[i]) % mod2
        # 存储一个编码组合是否出现过
        seen = {(h1, h2)}
        for start in range(1, n - m + 1):
            h1 = (h1 * a1 - arr[start - 1] * aL1 + arr[start + m - 1]) % mod1
            h2 = (h2 * a2 - arr[start - 1] * aL2 + arr[start + m - 1]) % mod2
            # 如果重复，则返回重复串的起点
            if (h1, h2) in seen:
                return start
            seen.add((h1, h2))
        # 没有重复，则返回-1
        return -1

s = Solution()

# print(s.longestDupSubstring(node_type_str))


# abcdebcdbcdbcbcfdbcdbcdbcfdbcdbcdbcfdbcdbcdbcbcfdbcdbcdbcfdbcdbcdbcfdbcdbcdbcfdbcdbcdbcbcfdbcdbcdbcfdbcdbcdbcfdbcdbcdbcfdbcdbcdbcfdbcdbcdbcfdbcdbcdbcbcfdbcdbcdbcfdbcdbcdbcfdghij

# bcfdbcdbcdbcfdbcdbcdbcfdbcdbcdbcbcfdbcdbcdbcfdbcdbcdbcfdbcdbcdbcfdbcdbcdbc

def find(s):

    l = len(s)

    for i in range(int(l/2), l):

        for j in range(i):

            sub_s = s[j :l-i+j+1]

            remain_s = s[l-i+j+1:]
            
            if sub_s in remain_s:
                # print()

                print(sub_s)

                # print(node_list[l-i+j].name)
            
                j = j + l-i

                import pdb; pdb.set_trace()

find(node_type_str)
# s = node_type_str
# print(s[len(s)])