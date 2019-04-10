class Node:
    def __init__(self, index, value):
        self.index = index
        self.value = value
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def get_child(self, index):
        return self.children[index]


def get_edges(root):
    edges = []
    stack = [root]
    while stack != []:
        current = stack.pop(0)
        for child in current.children:
            edges.append((current.index, child.index))
            stack.append(child)
    return edges


def get_labels(root):
    labels = []
    stack = [root]
    while stack != []:
        current = stack.pop(0)
        labels.append((current.index, {"label": current.value}))
        for child in current.children:
            stack.append(child)
    return labels


def build_tree(string, node_type=int):
    index = 0
    lst = eval(string)

    try:
        value, subtree = lst
    except ValueError:
        value, subtree = lst, []

    root = Node(index, value)
    subtree_stack = [subtree]
    parent_stack = [root]
    while parent_stack != []:
        tree = parent_stack.pop(0)
        subtree = subtree_stack.pop(0)
        for value in subtree:
            if isinstance(value, node_type):
                index = index + 1
                tree.add_child(Node(index, value))
            if isinstance(value, list):
                parent_stack.append(tree.get_child(-1))
                subtree_stack.append(value)
    return root
