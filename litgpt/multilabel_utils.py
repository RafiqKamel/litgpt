import networkx as nx
import random
import regex as re
import torch

stop_token = "%"
split_token = "%SPLIT%"
end_of_node_token = "$"


def recreate_graph(edge_list_str: str, nodes):
    if edge_list_str == "":
        graph = nx.DiGraph()
        graph.add_node(0)
        return graph
    # Split the string into lines
    edge_list_lines = edge_list_str.strip().split("\n")

    # Create a list of edge tuples
    edges = [tuple(map(int, line.split())) for line in edge_list_lines]

    # Create a new graph and add the edges
    G_loaded = nx.DiGraph()
    G_loaded.add_edges_from(edges)
    # add labels to nodes
    for i, node in enumerate(nodes):
        G_loaded.nodes[i]["label"] = node
    return G_loaded


def find_roots(G):
    roots = [node for node, in_degree in G.in_degree() if in_degree == 0]
    return roots


def tokenize_graph_labels(g, tokenizer):
    g = g.copy()
    for i, node in g.nodes(data=True):

        label = node["label"]
        tokens = (
            tokenize(tokenizer=tokenizer, target=label) + [end_of_node_token]
            if label != stop_token
            else label
        )
        node["tokens"] = tokens
    return g


def expand_list_nodes(G):
    """Transforms a tree where nodes contain lists into a tree where each element in the list is a separate node in a chain."""
    new_G = nx.DiGraph()
    node_mapping = {}  # Maps original nodes to their transformed first/last nodes

    for node, data in G.nodes(data=True):
        value = data.get("tokens")

        if isinstance(value, list) and value:
            # Create chain of nodes for list values
            prev_node = None
            first_new_node = None
            last_new_node = None

            for i, element in enumerate(value):
                new_node = f"{node}_{i}"  # Unique name
                new_G.add_node(new_node, value=element)

                if prev_node is not None:
                    new_G.add_edge(prev_node, new_node)  # Chain them
                else:
                    first_new_node = new_node  # Store first element

                prev_node = new_node  # Update last node
                last_new_node = new_node

            # Store mapping for edge redirection
            node_mapping[node] = (first_new_node, last_new_node)

        else:
            # Keep nodes without lists as they are
            new_G.add_node(node, value=value)
            node_mapping[node] = (node, node)  # Map to itself

    # Copy and redirect edges
    for u, v in G.edges():
        u_start, u_end = node_mapping[u]  # First and last node of transformed u
        v_start, _ = node_mapping[v]  # First node of transformed v

        new_G.add_edge(u_end, v_start)  # Redirect edges

    return new_G


def merge_duplicate_children(G, label_attr="label"):
    """
    Merges nodes in a directed graph G where a parent has multiple children with the same label.

    :param G: A directed graph (DiGraph) that resembles a tree (possibly with two roots).
    :param label_attr: The node attribute that contains the label.
    :return: A modified graph with duplicate children merged.
    """
    new_G = nx.DiGraph()
    new_G.add_nodes_from(G.nodes(data=True))  # Copy all nodes first
    removed_nodes = set()  # Track nodes that have been merged
    duplicated_children = 1  # Track nodes that have been merged
    node_mapping = {}  # Maps original nodes to their transformed first/last nodes
    roots = find_roots(G)
    new_G.add_node("ROOT")
    new_G.nodes["ROOT"][label_attr] = ""
    for root in roots:
        new_G.add_edge("ROOT", root)
    while duplicated_children:
        duplicated_children = 0
        removed_nodes = set()
        for parent in list(G.nodes):  # Ensure we're iterating over original nodes
            if parent in removed_nodes:
                continue
            children = list(G.successors(parent))
            label_to_node = {}  # Tracks the first occurrence of each label

            for child in children:
                label = G.nodes[child].get(label_attr, None)  # Get label of child

                if label in label_to_node:
                    # Found a duplicate-labeled child, merge it
                    merged_node = label_to_node[label]
                    duplicated_children += 1
                    # Redirect all edges from the duplicate child to merged_node
                    for grandchild in G.successors(child):
                        new_G.add_edge(merged_node, grandchild)

                    # Remove the duplicate child from the graph
                    if new_G.has_node(child):
                        new_G.remove_node(child)
                        removed_nodes.add(child)
                        node_mapping[child] = merged_node
                else:
                    # First occurrence of this label, keep it
                    label_to_node[label] = child
                if child not in removed_nodes:
                    new_G.add_edge(parent, child)  # Retain the edge
        G = new_G.copy()
    for n in new_G.nodes:
        if n not in node_mapping:
            node_mapping[n] = n
    return new_G, node_mapping


def find_path_from_labels(graph: nx.Graph, sequence_labels: list, roots: list) -> list:
    """
    Finds a path in the given graph that follows the specified sequence of labels, starting from one of the specified roots.

    :param graph: A NetworkX graph where each edge has a 'label' attribute.
    :param labels: A list of labels representing the desired path sequence.
    :param roots: A list of nodes from which the search should start.
    :return: A list of nodes representing the path, or an empty list if no path is found.
    """

    def dfs(node_list, label_idx, path):
        node = node_list[-1]
        if label_idx == len(sequence_labels):
            return path, node_list

        for neighbor in graph.neighbors(node):
            if (
                "value" in graph.nodes[neighbor]
                and graph.nodes[neighbor]["value"] == sequence_labels[label_idx]
            ):
                new_value = graph.nodes[neighbor]["value"]
                new_path = path + [new_value]
                node_list.append(neighbor)
                result, _ = dfs(node_list, label_idx + 1, new_path)
                if result:
                    return result, node_list

        return None, None

    # Try starting from one of the specified root nodes
    for start_node in roots:
        if start_node in graph:
            start_value = graph.nodes[start_node].get("value", None)
            result, node_list = dfs([start_node], 1, [start_value])
            if result:
                return node_list

    raise ValueError("No path found")


def get_values_of_children(graph, node):
    return [graph.nodes[child]["value"] for child in graph.successors(node)]


def remove_child_with_value(graph, node, value):
    for child in list(graph.successors(node)):
        if graph.nodes[child]["value"] == value:
            graph.remove_edge(node, child)
            return True
    return False


def random_dfs(graph: nx.Graph, roots: list):
    visited = set()
    traversal = []

    def dfs(node):
        stack = [node]
        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                traversal.append((graph.nodes[current]["value"], current))
                neighbors = list(graph.neighbors(current))
                neighbors.sort(
                    key=lambda x: graph.nodes[x]["value"] == stop_token, reverse=True
                )  # Push stop_token to the end
                random.shuffle(neighbors[:-1])  # Shuffle all except stop_token
                stack.extend(neighbors)

    random.shuffle(roots)  # Shuffle root order to ensure randomness
    for root in roots:
        if root not in visited:
            dfs(root)

    return traversal


def is_pattern_broken(accumulated, pattern):
    """
    Returns True if the accumulated string can no longer be extended
    to match the expected pattern.
    """
    # Try to match the accumulated string, allowing partial matches.
    match = pattern.match(accumulated, partial=True)
    # If no match exists at all, the pattern is broken.
    return match is None


def update_pointer_indices_in_sequence_and_parsing_graph(
    sequence, parsing_graph, node_mapping
):
    pointer_map = {}
    updated_sequence = []
    new_index = 0
    token_buffer = []
    expecting_pointer = False  # Tracks if we are attempting to form a pointer
    expected_pattern = r"<pointer:(\d+).*"
    for token in sequence:
        if token[0] == "<" or token[0].startswith("<"):
            expecting_pointer = True
        if expecting_pointer:
            token_buffer.append(token[0])
            # check if the pattern is broken and reset the buffer
            if is_pattern_broken("".join(token_buffer), re.compile(expected_pattern)):
                token_buffer = []
                expecting_pointer = False
        buffer_str = "".join(token_buffer)
        pointer_match = re.match(expected_pattern, buffer_str)
        if pointer_match:
            pointer_index = int(pointer_match.group(1))
            expecting_pointer = False
            token_buffer = []
            if pointer_index not in pointer_map:
                pointer_map[pointer_index] = new_index
                new_index += 1
            token_to_add = token[0].replace(
                str(pointer_index), str(pointer_map[pointer_index])
            )
            parsing_graph.nodes[node_mapping[token[1]]]["value"] = token_to_add
            updated_sequence.append([token_to_add, token[1], True])
        else:
            updated_sequence.append(list(token) + [False])
    return updated_sequence


def return_all_possibilities_for_sequence(
    parsing_tree, sequence, root_values, node_mapping
):
    parsing_tree = parsing_tree.copy()
    possibilites_sequence = []
    seen_pointers = set()
    starting_possibilities = (sequence[0][0], root_values)
    if stop_token in starting_possibilities[1]:
        starting_possibilities[1].remove(stop_token)
    starting_possibilities = list(
        set([starting_possibilities[0]] + starting_possibilities[1])
    )
    possibilites_sequence.append(starting_possibilities)
    start_index = 0
    for i in range(1, len(sequence)):
        if sequence[i - 1] == stop_token:
            start_index = i
            continue
        sequence_part = sequence[start_index:i]
        associated_node = node_mapping[sequence_part[-1][1]]
        possibilities = [
            sequence[i][0],
            get_values_of_children(parsing_tree, associated_node),
        ]
        if sequence[i][2]:
            seen_pointers.add(sequence[i][0])
            print("Pointer detected", seen_pointers, possibilities[1])
            possibilities[1] = list(set(possibilities[1]) & seen_pointers)
            print("After filtering", possibilities[1])
        if stop_token in possibilities[1]:
            possibilities[1].remove(stop_token)
        possibilities = list(set([possibilities[0]] + possibilities[1]))
        possibilites_sequence.append(possibilities)
        next_node_value = sequence[i][0]
        remove_child_with_value(parsing_tree, associated_node, next_node_value)
    return possibilites_sequence


def encode_amr_sequence_and_possibilities(sequence, possibilities, tokenizer):
    encoded_sequence = [encode_without_special_chars(tokenizer, x[0]) for x in sequence]
    for tok in encoded_sequence:
        if len(tok) != 1:
            raise ValueError("Tokenization error")
    encoded_possibilities = [
        [
            encode_without_special_chars(tokenizer=tokenizer, target=x)[0]
            for x in token_list
        ]
        for token_list in possibilities
    ]
    encoded_sequence = torch.tensor(encoded_sequence)
    return encoded_sequence, encoded_possibilities


def prepare_sequence_and_possibilities(amr_linearization, graph_structure, tokenizer):
    amr_linearization = amr_linearization.replace("<stop>", stop_token)
    sequence = amr_linearization.split(split_token)
    graph = recreate_graph(graph_structure, sequence)
    tokenized_graph = tokenize_graph_labels(graph, tokenizer)
    expanded_graph = expand_list_nodes(tokenized_graph)
    parse_tree, node_mapping = merge_duplicate_children(
        expanded_graph, label_attr="value"
    )
    roots = find_roots(expanded_graph)
    sequence = random_dfs(expanded_graph, roots)
    sequence_updated = update_pointer_indices_in_sequence_and_parsing_graph(
        sequence, parse_tree, node_mapping
    )
    all_possibilites = return_all_possibilities_for_sequence(
        parse_tree,
        sequence_updated,
        [expanded_graph.nodes[root]["value"] for root in roots],
        node_mapping,
    )
    encode_amr_sequence_and_possibilities(sequence_updated, all_possibilites, tokenizer)
    return sequence_updated, all_possibilites


def tokenize(tokenizer, target):
    filtered_tokens = encode_without_special_chars(tokenizer=tokenizer, target=target)
    return [tokenizer.decode(token_id) for token_id in filtered_tokens]


def encode_without_special_chars(tokenizer, target):
    special_tokens = [0, 2]
    encoded = tokenizer.encode(target)
    filtered_tokens = [
        token_id for token_id in encoded if token_id not in special_tokens
    ]
    if len(filtered_tokens) == 0:
        return encoded[:1]
    return filtered_tokens


def extract_pointer_index(token):
    token = token.replace(" ", "")
    pattern = r"<pointer:(\d+)>"
    match = re.match(pattern, token)
    if match:
        return int(match.group(1))
    raise ValueError(f"Token {token} is not a pointer")


def extract_pointer_name(token):
    token = token.replace(" ", "")
    pattern = r"<pointer:\d+>(.*)"
    match = re.match(pattern, token)
    if match:
        return match.group(1)
    raise ValueError(f"Token {token} is not a pointer")


def check_if_pointer(token):
    pattern = r"<pointer:\d+>.*"
    match = re.match(pattern, token)
    return match is not None


def create_variable_name(pointer_name, current_variables):
    # get first letter
    first_letter = pointer_name[0]
    if first_letter not in current_variables:
        return first_letter
    else:
        for i in range(1, 26):
            if first_letter + str(i) not in current_variables:
                return first_letter + str(i)
        raise ValueError("Too many variables")


def delinearize_into_triples(linearization):
    structures = linearization.split(stop_token)
    pointer_index_to_name = {}
    pointer_index_to_variable = {}
    current_variables = set()
    triples = []
    for structure in structures:
        nodes = structure.split(end_of_node_token)
        while "" in nodes:
            nodes.remove("")
        while " " in nodes:
            nodes.remove(" ")
        if len(nodes) == 0:
            continue
        main_pointer = nodes[0]
        if not check_if_pointer(main_pointer):
            raise ValueError(f"Main pointer {main_pointer} is not a pointer")
        pointer_index = extract_pointer_index(main_pointer)
        pointer_name = extract_pointer_name(main_pointer)
        if pointer_index not in pointer_index_to_variable:
            main_variable = create_variable_name(pointer_name, current_variables)
            pointer_index_to_variable[pointer_index] = main_variable
            current_variables.add(main_variable)
        else:
            main_variable = pointer_index_to_variable[pointer_index]
        triples.append((main_variable, ":instance", pointer_name))
        current_edge = None
        for node in nodes[1:]:
            if not current_edge:
                if not node.replace(" ", "").startswith(":"):
                    raise ValueError(
                        f"Node ({node}) is not an edge and there is no current edge",
                        f"main_variable: {main_variable}",
                        f"structure: {structure}",
                        f"nodes: {nodes}",
                    )
                else:
                    current_edge = node
            else:
                if node.replace(" ", "").startswith(":"):
                    raise ValueError(
                        f"Node ({node}) is an edge, two edges in a row",
                        f"current_edge: {current_edge}",
                        f"main_variable: {main_variable}",
                        f"structure: {structure}",
                    )
                if not check_if_pointer(node):
                    triples.append((main_variable, current_edge, node))
                    current_edge = None
                else:
                    pointer_index = extract_pointer_index(node)
                    pointer_name = extract_pointer_name(node)
                    if pointer_index not in pointer_index_to_name:
                        pointer_index_to_name[pointer_index] = pointer_name
                        if pointer_index not in pointer_index_to_variable:
                            pointer_index_to_variable[pointer_index] = (
                                create_variable_name(pointer_name, current_variables)
                            )
                            current_variables.add(
                                pointer_index_to_variable[pointer_index]
                            )
                        triples.append(
                            (
                                pointer_index_to_variable[pointer_index],
                                ":instance",
                                pointer_name,
                            )
                        )
                        triples.append(
                            (
                                main_variable,
                                current_edge,
                                pointer_index_to_variable[pointer_index],
                            )
                        )
                    else:
                        triples.append(
                            (
                                main_variable,
                                current_edge,
                                pointer_index_to_variable[pointer_index],
                            )
                        )
                    current_edge = None
    return list(set(triples))


from smatch import amr as smatch_amr


def triples_to_smatch_AMR(triples, postfix=""):
    node_list = []
    node_to_index = {}
    node_value_list = []
    for triple in triples:
        if triple[1] == ":instance":
            node_list.append(triple[0] + postfix)
            node_value_list.append(triple[2])
            node_to_index[triple[0] + postfix] = len(node_list) - 1
    edge_list = []
    attribute_list = []
    index_to_edges = {}
    index_to_attributes = {}
    for triple in triples:
        index = node_to_index.get(triple[0] + postfix)
        if index is not None:
            if triple[1] == ":instance":
                continue
            if triple[2] + postfix in node_list:
                if index in index_to_edges:
                    index_to_edges[index].append([triple[1], triple[2] + postfix])
                else:
                    index_to_edges[index] = [[triple[1], triple[2] + postfix]]
            else:
                if index in index_to_attributes:
                    index_to_attributes[index].append([triple[1], triple[2]])
                else:
                    index_to_attributes[index] = [[triple[1], triple[2]]]

    for i in range(len(node_list)):
        if i in index_to_edges:
            edge_list.append(index_to_edges[i])
        else:
            edge_list.append([])
        if i in index_to_attributes:
            attribute_list.append(index_to_attributes[i])
        else:
            attribute_list.append([])
    return smatch_amr.AMR(
        node_list=node_list,
        node_value_list=node_value_list,
        relation_list=edge_list,
        attribute_list=attribute_list,
    )


from smatch import get_best_match, compute_f
import smatch


def get_amr_match(
    amr1, amr2, justinstance=False, justattribute=False, justrelation=False
):
    prefix1 = "a"
    prefix2 = "b"
    # Rename node to "a1", "a2", .etc
    amr1.rename_node(prefix1)
    # Renaming node to "b1", "b2", .etc
    amr2.rename_node(prefix2)
    (instance1, attributes1, relation1) = amr1.get_triples()
    (instance2, attributes2, relation2) = amr2.get_triples()
    # optionally turn off some of the node comparison
    doinstance = doattribute = dorelation = True
    if justinstance:
        doattribute = dorelation = False
    if justattribute:
        doinstance = dorelation = False
    if justrelation:
        doinstance = doattribute = False
    (best_mapping, best_match_num) = get_best_match(
        instance1,
        attributes1,
        relation1,
        instance2,
        attributes2,
        relation2,
        prefix1,
        prefix2,
        doinstance=doinstance,
        doattribute=doattribute,
        dorelation=dorelation,
    )
    if justinstance:
        test_triple_num = len(instance1)
        gold_triple_num = len(instance2)
    elif justattribute:
        test_triple_num = len(attributes1)
        gold_triple_num = len(attributes2)
    elif justrelation:
        test_triple_num = len(relation1)
        gold_triple_num = len(relation2)
    else:
        test_triple_num = len(instance1) + len(attributes1) + len(relation1)
        gold_triple_num = len(instance2) + len(attributes2) + len(relation2)
    return best_match_num, test_triple_num, gold_triple_num


def score_amr_pairs(
    amrs1, amrs2, justinstance=False, justattribute=False, justrelation=False
):
    """
    Score one pair of AMR lines at a time from each file handle
    :param f1: file handle (or any iterable of strings) to read AMR 1 lines from
    :param f2: file handle (or any iterable of strings) to read AMR 2 lines from
    :param justinstance: just pay attention to matching instances
    :param justattribute: just pay attention to matching attributes
    :param justrelation: just pay attention to matching relations
    :return: generator of cur_amr1, cur_amr2 pairs: one-line AMR strings
    """
    # matching triple number, triple number in test file, triple number in gold file
    total_match_num = total_test_num = total_gold_num = 0
    single_score = True
    # Read amr pairs from two files
    for cur_amr1, cur_amr2 in zip(amrs1, amrs2):
        best_match_num, test_triple_num, gold_triple_num = get_amr_match(
            cur_amr1,
            cur_amr2,
            justinstance=justinstance,
            justattribute=justattribute,
            justrelation=justrelation,
        )
        total_match_num += best_match_num
        total_test_num += test_triple_num
        total_gold_num += gold_triple_num
        # clear the matching triple dictionary for the next AMR pair
        smatch.match_triple_dict.clear()
        if (
            not single_score
        ):  # if each AMR pair should have a score, compute and output it here
            yield compute_f(best_match_num, test_triple_num, gold_triple_num)
    if (
        single_score
    ):  # output document-level smatch score (a single f-score for all AMR pairs in two files)
        return compute_f(total_match_num, total_test_num, total_gold_num)[2]
