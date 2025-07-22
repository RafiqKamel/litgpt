import networkx as nx
import re


def parse_bfs_to_digraph(bfs_list):
    """
    Parses a BFS-formatted list into a NetworkX directed graph.

    Args:
        bfs_list (list): A BFS-formatted list of nodes and edges.

    Returns:
        nx.DiGraph: A directed graph representing the structure of the input list.
    """
    graph = nx.DiGraph()
    queue = []  # Queue to track the BFS order
    current_node = None

    i = 0
    while i < len(bfs_list):
        item = bfs_list[i]

        if isinstance(item, str) and not item == "<stop>":
            # Add the node to the graph
            node_id = item
            graph.add_node(i, label=node_id)

            if current_node is None:
                current_node = node_id
                current_node_index = i
            else:
                queue.append(node_id)  # Add to the BFS queue

        elif isinstance(item, str) or item[1].startswith(":"):
            if isinstance(item, tuple) and item[1].startswith(":"):
                edge_property = item
                next_node = bfs_list[i + 1]
                if isinstance(next_node, str):
                    target_id = next_node
                    graph.add_edge(current_node_index, i + 1, label=edge_property)
                    graph.nodes[i + 1]["label"] = target_id
                    queue.append(target_id)
                i += 1  # Skip the next node since it's already processed
            elif item == "<stop>":
                # End of this branch; move to the next node in the queue
                current_node = None
                queue = []

        i += 1

    return graph


def preprocess_bfs_pointers(bfs_trace):
    pointer_map = {}
    indexed_bfs_trace = []
    index = 0
    for item in bfs_trace:
        if item.startswith("<pointer:"):
            match = re.search(r"<pointer:(\d+)>", item)
            pointer_index = int(match.group(1))
            if pointer_index not in pointer_map:
                last_token_is_pointer = True
                pointer_token = item
                continue

        if item == "<s>" or item == "</s>":
            continue
        if item == "<stop>":
            indexed_bfs_trace.append(item)
            index += 1
        else:
            if last_token_is_pointer:
                item = pointer_token + " " + item
                pointer_map[pointer_index] = item
                last_token_is_pointer = False
            if item.startswith(":"):
                indexed_bfs_trace.append((index, item))
            else:
                indexed_bfs_trace.append(item)
            index += 1
    return indexed_bfs_trace, pointer_map


def transform_graph_with_edge_labels(G):
    # Create a new directed graph for the transformed version
    H = nx.DiGraph()
    nodes_label_map = {data[0]: data[1] for node, data in enumerate(G.nodes(data=True))}
    # Create nodes in H from original nodes in G
    H.add_nodes_from(G.nodes)
    for node, data in G.nodes(data=True):
        H.nodes[node]["label"] = nodes_label_map[node]["label"]

    # Edge transformation: source, target, label -> source, target and new edge label node
    for u, v, w in G.edges(data=True):
        label_node = w["label"][1]
        label_index = w["label"][0]
        u_label = nodes_label_map[u]["label"]
        v_label = nodes_label_map[v]["label"]
        H.add_edge(u, label_index)
        H.add_edge(label_index, v)
        H.nodes[label_index]["label"] = label_node
        H.nodes[u]["label"] = u_label
        H.nodes[v]["label"] = v_label
    return H


def rearrange_graph(G):
    visited = set()  # Keep track of visited nodes
    new_graph = nx.DiGraph()  # Directed graph to store rearranged structure
    node_mapping = {}  # Map original nodes to new graph node indices
    node_index = 0  # Unique index for nodes in the new graph
    for start_node in G.nodes:
        if start_node not in visited:
            # Perform BFS for this connected component
            queue = [start_node]
            visited.add(start_node)
            root_node = start_node

            while queue:
                current_node = queue.pop(0)

                # Add the current node to the new graph if not already added
                if current_node not in node_mapping:
                    label = G.nodes[current_node].get("label", current_node)
                    new_graph.add_node(node_index, label=label)
                    node_mapping[current_node] = node_index
                    current_node_index = node_index
                    if current_node == root_node:
                        root_node_index = node_index
                    node_index += 1
                else:
                    current_node_index = node_mapping[current_node]

                # Process neighbors
                for neighbor in G.neighbors(current_node):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

                    # Add the neighbor to the new graph
                    label = G.nodes[neighbor].get("label", neighbor)
                    new_graph.add_node(node_index, label=label)
                    node_mapping[neighbor] = node_index
                    new_graph.add_edge(current_node_index, node_index)
                    node_index += 1

            # Add <stop> node after processing all neighbors
            new_graph.add_node(node_index, label="<stop>")
            new_graph.add_edge(root_node_index, node_index)
            node_index += 1

    return new_graph


def create_pointer_map(graph):
    pointer_map = {}
    # Add a capturing group to extract the number
    pattern = r"<pointer:(\d+)>"

    for node, data in graph.nodes(data=True):
        text = data["label"]
        text = str(text)
        match = re.search(pattern, text)
        if match:
            # Extract the pointer number from the first capturing group
            pointer_index = int(match.group(1))
            if pointer_index not in pointer_map:
                pointer_map[pointer_index] = [node]
            else:
                pointer_map[pointer_index].append(node)

    return pointer_map


def duplicate_edges_for_same_pointer_nodes(graph, pointer_map):
    edges = []
    in_edges = []
    for pointer_index, nodes in pointer_map.items():
        if len(nodes) > 1:
            for i in nodes:
                for j in nodes:
                    if i == j:
                        continue
                    # get all edges from node i
                    curr_edges = list(graph.edges(i))
                    for edge in curr_edges:
                        edges.append((j, edge[1]))

                    curr_in_edges = list(graph.in_edges(nbunch=i))
                    for edge in curr_in_edges:
                        in_edges.append((edge[0], j))
    for edge in edges:
        graph.add_edge(edge[0], edge[1])
    for edge in in_edges:
        graph.add_edge(edge[0], edge[1])
    return graph


# Define DFS function that works for disconnected graphs
def dfs_traversal(graph):
    visited = set()
    dfs_result = []
    dfs_global = []

    def dfs(node):
        visited.add(node)
        dfs_result.append(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                dfs(neighbor)

    for node in graph.nodes():
        if node not in visited:
            dfs(node)
            dfs_global += dfs_result
            dfs_result = []  # Reset for next component
    return dfs_global


def replace_pointer_references_with_full_labels(bfs_trace, pointer_map):
    for i, item in enumerate(bfs_trace):
        if isinstance(item, str) and item.startswith("<pointer:"):
            match = re.search(r"<pointer:(\d+)>", item)
            pointer_index = int(match.group(1))
            full_label = pointer_map[pointer_index]
            bfs_trace[i] = full_label
    return bfs_trace


def preprocess_spring_linearization(
    tokens,
    split_token="%SPLIT%",
    duplicate_edges=True,
    duplicate_pointer_labels_for_all_pointers=False,
):
    ## for AMR to text we should keep duplicate edges True as we want to keep the structure. for text to AMR we should set it to False as we want to remove the duplicate edges to have seperate structures

    indexed_bfs_trace, pointer_to_label_map = preprocess_bfs_pointers(tokens)
    if duplicate_pointer_labels_for_all_pointers:
        indexed_bfs_trace = replace_pointer_references_with_full_labels(
            indexed_bfs_trace, pointer_to_label_map
        )
    graph = parse_bfs_to_digraph(indexed_bfs_trace)

    graph = transform_graph_with_edge_labels(graph)

    graph = rearrange_graph(graph)

    dfs = dfs_traversal(graph)
    dfs_string = ""
    for node in dfs:
        dfs_string += str(graph.nodes[node]["label"]) + " "
    dfs_string = split_token.join([str(graph.nodes[node]["label"]) for node in dfs])
    dfs_string_compare = " ".join([str(graph.nodes[node]["label"]) for node in dfs])
    remapping = {n: i for i, n in enumerate(dfs)}
    graph = nx.relabel_nodes(graph, remapping)
    pointer_map = create_pointer_map(graph)
    if duplicate_edges:
        graph = duplicate_edges_for_same_pointer_nodes(graph, pointer_map)
    instruction = " ".join(tokens)
    linearized = split_token.join(
        [graph.nodes[n]["label"] for n in sorted(graph.nodes())]
    )
    linearized_compare = " ".join(
        [graph.nodes[n]["label"] for n in sorted(graph.nodes())]
    )

    correct = dfs_string_compare == instruction and linearized_compare == instruction
    if not correct:
        print(instruction)
        print(dfs_string_compare)
        print(linearized_compare)
        print("**********")
    return correct, graph, dfs_string

from spring_amr.linearization import AMRLinearizer
lin = AMRLinearizer(use_pointer_tokens=True)
def preprocess_amr(amr):
        l = lin.linearize(amr)
        nodes = l.nodes
        nodes.remove("<s>")
        nodes.remove("</s>")
        _, graph, dfs_string = preprocess_spring_linearization(nodes, split_token="%SPLIT%")
        graph_str = create_graph_str_from_nx(graph)
        print("dfs string:",dfs_string)
        return dfs_string, graph_str

def create_graph_str_from_nx(graph):
    graph_str = ""
    for u, v in graph.edges:
        graph_str += f"{u} {v}\n"
    return graph_str