import math
import graphviz

# calculate entropy
def entropy(data):
    total = len(data)
    label_counts = {}
    
    for row in data:
        label = row[-1]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    ent = sum(- (count / total) * math.log2(count / total) for count in label_counts.values())
    return ent

# calculate info gain
def info_gain(data, split_attr_index):
    total_entropy = entropy(data)
    values = set(row[split_attr_index] for row in data)
    
    weighted_entropy = sum((len(subset) / len(data)) * entropy(subset) for value in values if (subset := [row for row in data if row[split_attr_index] == value]))
    
    return total_entropy - weighted_entropy

# build the decision tree
def build_tree(data, attributes):
    labels = [row[-1] for row in data]
    
    if len(set(labels)) == 1:
        return labels[0]
    
    if not attributes:
        return max(set(labels), key=labels.count)
    
    gains = [info_gain(data, i) for i in range(len(attributes))]
    best_attr_index = gains.index(max(gains))
    best_attr = attributes[best_attr_index]
    
    tree = {best_attr: {}}
    
    for value in set(row[best_attr_index] for row in data):
        subset = [row for row in data if row[best_attr_index] == value]
        new_attrs = attributes[:best_attr_index] + attributes[best_attr_index+1:]
        tree[best_attr][value] = build_tree([row[:best_attr_index] + row[best_attr_index+1:] for row in subset], new_attrs)
    
    return tree

# classify new instances
def classify(tree, attributes, sample):
    while isinstance(tree, dict):
        root_attr = next(iter(tree))
        root_attr_index = attributes.index(root_attr)
        value = sample[root_attr_index]
        tree = tree[root_attr].get(value, None)
        if tree is None:
            return max(set(row[-1] for row in data), key=lambda cls: [row[-1] for row in data].count(cls))
    return tree

# decision rules
def extract_rules(tree, attributes, rule="", rules=[]):
    if isinstance(tree, str):
        rules.append(rule.strip() + " -> " + tree)
        return
    
    root_attr = next(iter(tree))
    for value, subtree in tree[root_attr].items():
        extract_rules(subtree, attributes, rule + f"{root_attr}={value} ", rules)
    
    return rules

# dataset
data = [
    ["sunny", 85, 85, False, "Don't Play"],
    ["sunny", 80, 90, True, "Don't Play"],
    ["overcast", 83, 78, False, "Play"],
    ["rain", 70, 96, False, "Play"],
    ["rain", 68, 80, False, "Play"],
    ["rain", 65, 70, True, "Don't Play"],
    ["overcast", 64, 65, True, "Play"],
    ["sunny", 72, 95, False, "Don't Play"],
    ["sunny", 69, 70, False, "Play"],
    ["rain", 75, 80, False, "Play"],
    ["sunny", 75, 70, True, "Play"],
    ["overcast", 72, 90, True, "Play"],
    ["overcast", 81, 75, False, "Play"],
    ["rain", 71, 80, True, "Don't Play"]
]
attributes = ["Outlook", "Temperature", "Humidity", "Windy"]

tree = build_tree(data, attributes)
print("\nDecision Tree:")
print(tree)

def visualize_tree(tree, parent_name='', graph=None):
    if graph is None:
        graph = graphviz.Digraph(format='png')
        graph.node('root', label=next(iter(tree)))
        visualize_tree(tree[next(iter(tree))], 'root', graph)
        return graph
    
    if isinstance(tree, dict):
        for value, subtree in tree.items():
            node_name = f'{parent_name}_{value}'
            graph.node(node_name, label=str(value))
            graph.edge(parent_name, node_name)
            if isinstance(subtree, dict):
                visualize_tree(subtree, node_name, graph)
            else:
                leaf_name = f'{node_name}_leaf'
                graph.node(leaf_name, label=subtree, shape='box')
                graph.edge(node_name, leaf_name)
    return graph

graph = visualize_tree(tree)
graph.render('decision_tree', view=True)

rules = extract_rules(tree, attributes, "", [])
print("\nInductive Rules:")
print("-------------------------------------------------")
for rule in rules:
    print(" " + rule)
print("-------------------------------------------------")

test_samples = [
    ["overcast", 63, 70, False],
    ["rain", 73, 90, True],
    ["sunny", 70, 73, True]
]

print("\nClassifications:")
print("-------------------------------------------------")
print("|  Sample Data\t\t\t|  Result\t|")
print("-------------------------------------------------")
for sample in test_samples:
    result = classify(tree, attributes, sample)
    print(f"|  {str(sample):25}\t|  {result:10}\t|")
print("-------------------------------------------------")