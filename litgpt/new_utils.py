import networkx as nx
import numpy as np
from litgpt.positional_encodings_config import (
    sinousidial_encodings_q,
    sinousidial_encodings_dim,
)
from litgpt.magentic_laplacian_utils import (
    magnetic_laplacian,
    magnetic_laplacian_eigenvectors,
)
import torch
import yaml
import unicodedata


def recreate_graph(edge_list_str: str):
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
    return G_loaded


def load_properties_from_yaml(file_path):
    """
    Load properties from a YAML file.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        dict: A dictionary containing the properties from the YAML file.
    """
    try:
        with open(file_path, "r") as file:
            properties = yaml.safe_load(file)
            return properties
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None
    except yaml.YAMLError as e:
        print(f"Error: The file {file_path} is not a valid YAML file. {e}")
        return None


def add_prefix_to_dict_keys(dict, prefix):
    """
    Add a prefix to all keys in a dictionary.

    Args:
        dict (dict): The dictionary to which to add the prefix.

    Returns:
        dict: The dictionary with the prefix added to all keys.
    """
    return {f"{prefix}{key}": value for key, value in dict.items()}


def xavier_initialization(shape):
    return torch.tensor(np.random.randn(*shape) * np.sqrt(2 / (shape[0] + shape[1])))


def positional_encoding(
    index, d_model=sinousidial_encodings_dim, q=sinousidial_encodings_q
):
    """
    Generate the sinusoidal positional encoding for a specific position in a sequence.

    Parameters:
    - index (int): The position in the sequence for which the encoding is being generated.
    - d_model (int): The dimensionality of the model.
    - q (float): The base of the exponent used in the positional encoding formula (default is 10000).

    Returns:
    - np.ndarray: The positional encoding for the given index.
    """
    encoding = np.zeros(d_model)

    for i in range(0, d_model, 2):
        angle = index / np.power(q, (2 * (i // 2)) / d_model)
        encoding[i] = np.sin(angle)
        if i + 1 < d_model:
            encoding[i + 1] = np.cos(angle)

    return encoding


def create_indexing_map(sentence: str, tokenizer):
    tokens = sentence.split()
    ids = tokenizer.encode(sentence)
    subtokens = [tokenizer.decode(id) for id in ids]

    index_map = {}
    subtoken_index = 0

    for i, token in enumerate(tokens):
        subtokens_for_token = []
        token_length = 0
        current_subtoken = ""
        internal_subtoken_index = 0

        while token_length < len(strip_string(token)):
            subtokens_for_token.append(internal_subtoken_index)
            internal_subtoken_index += 1
            token_length += len(strip_string(subtokens[subtoken_index]))
            current_subtoken += subtokens[subtoken_index]
            subtoken_index += 1
        if strip_string(token) != strip_string(current_subtoken):
            raise ValueError(f"Tokenization mismatch: {token} != {current_subtoken}")
        index_map[i] = subtokens_for_token

    return index_map


def strip_string(string):
    string = unicodedata.normalize("NFC", string)
    return string.replace(" ", "")


def process_eigenvectors_subtokens(tokenizer, sentence, eigvecs):
    if len(eigvecs) == len(tokenizer.encode(sentence)):
        indexing_map = {i: [i] for i in range(len(eigvecs))}
    else:
        indexing_map = create_indexing_map(sentence, tokenizer)
    subtoken_eigvecs = []
    global_subtoken_index = 0
    for i, eigvec in enumerate(eigvecs):
        subtoken_indices = indexing_map[i]
        subtoken_eigvecs.extend([eigvec] * len(subtoken_indices))
        for subtoken_index in subtoken_indices:
            sinousoidal_encoding = positional_encoding(subtoken_index)

            # concatenate the eigenvector with the positional encoding
            subtoken_eigvecs[global_subtoken_index] = np.concatenate(
                (eigvec, sinousoidal_encoding)
            )
            global_subtoken_index += 1

    return np.array(subtoken_eigvecs)


def create_edge_list_sequence(n_tokens):
    edge_list = ""
    for i in range(n_tokens - 1):
        edge_list += f"{i} {i+1}\n"
    return edge_list


def prepare_eigvecs_datapoint(
    tokenizer, graph_str, sentence, prompt_style, max_seq_length
):
    G = recreate_graph(edge_list_str=graph_str)
    eigvecs = magnetic_laplacian_eigenvectors(g=G, max_seq_length=max_seq_length)
    subtoken_eigvecs = process_eigenvectors_subtokens(
        tokenizer=tokenizer, sentence=sentence, eigvecs=eigvecs
    )
    if type(prompt_style) == str:
        if prompt_style == "amr2text":
            starting_token_ids = tokenizer.encode("<AMR>")
        elif prompt_style == "text2amr":
            starting_token_ids = tokenizer.encode("<text>")
        else:
            starting_token_ids = []
            print("Error: Unknown prompt style", prompt_style)
    else:
        if prompt_style.name() == "amr2text":
            starting_token_ids = tokenizer.encode("<AMR>")
        elif prompt_style.name() == "text2amr":
            starting_token_ids = tokenizer.encode("<text>")
        else:
            starting_token_ids = []
            print("Error: Unknown prompt style", prompt_style)
    len_starting_token_ids = len(starting_token_ids)
    return subtoken_eigvecs, len_starting_token_ids, starting_token_ids
