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
import re


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
    encoding = encoding / torch.norm(torch.tensor(encoding))
    return encoding


def split_preserve_quotes(s):
    return re.findall(r'\".*?\"|\S+', s)

def create_indexing_map(sentence: str, tokenizer, num_of_nodes):
    ids = nodewise_tokenize(sentence, tokenizer)
    if num_of_nodes == len(ids):
        print("No need to process eigenvectors", flush=True)
        return {i: [i] for i in range(num_of_nodes)}
    if "%SPLIT%" in sentence:
        tokens = sentence.split("%SPLIT%")
        sentence = sentence.replace("%SPLIT%", " ")
    else:    
        tokens = split_preserve_quotes(sentence)
    if len(tokens) != num_of_nodes:
        print(tokens)
        print(num_of_nodes)
        raise ValueError(f"Number of nodes mismatch: {len(tokens)} != {num_of_nodes}")
        
    subtokens = [tokenizer.decode(id) for id in ids]

    index_map = {}
    subtoken_index = 0

    for i, token in enumerate(tokens):
        subtokens_for_token = []
        token_length = 0
        current_subtoken = ""
        internal_subtoken_index = 0
        token = token + " "
        while len(current_subtoken) < len(token):
            subtokens_for_token.append(internal_subtoken_index)
            internal_subtoken_index += 1
            token_length += len(subtokens[subtoken_index])
            current_subtoken += subtokens[subtoken_index]
            subtoken_index += 1
        if token != current_subtoken:
            raise ValueError(f"Tokenization mismatch: ({token}) != ({current_subtoken})")
        index_map[i] = subtokens_for_token

    return index_map




def process_eigenvectors_subtokens( eigvecs, num_of_nodes, indexing_map):
    if len(eigvecs) != num_of_nodes:
        raise ValueError(
            f"Number of eigenvectors ({len(eigvecs)}) does not match number of nodes ({num_of_nodes})"
        )
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
    graph_str, max_seq_length, num_of_eigenvecs, indexing_map
):
    G = recreate_graph(edge_list_str=graph_str)
    eigvecs = magnetic_laplacian_eigenvectors(g=G, max_seq_len=max_seq_length, num_of_eigenvecs=num_of_eigenvecs)
    subtoken_eigvecs = process_eigenvectors_subtokens(
        eigvecs=eigvecs, num_of_nodes=len(G.nodes), indexing_map=indexing_map
    )
    return subtoken_eigvecs

def starting_token_len(prompt_style, tokenizer):
    starting_token = prompt_style.starting_token()
    starting_token_ids = tokenizer.encode(starting_token)
    len_starting_token_ids = len(starting_token_ids)
    return len_starting_token_ids, starting_token_ids

def update_positional_mlp_lr(
    optimizer, model, new_lr, target_module_name="positional_encoding_mlp"
):
    updated_groups = 0  # Track updates for logging or debugging
    for name, param in model.named_parameters():
        if target_module_name in name:
            for param_group in optimizer.param_groups:
                # Correct way to check if `param` is in the parameter group
                if any(p is param for p in param_group["params"]):
                    #print(f"Setting LR for {name} in param group")
                    param_group["lr"] = new_lr
                    updated_groups += 1
                    break
        else:
            pass  # Reduce verbose output

    if updated_groups == 0:
        print(
            f"No parameter groups were updated. Check if {target_module_name} is correct."
        )


def mark_MLP_for_finetuning(model, target_module_name="positional_encoding_mlp"):
    for name, param in model.named_parameters():
        if target_module_name in name:
            print(f"Marking {name} for finetuning")
            param.requires_grad = True


def nodewise_tokenize(prompt,  tokenizer,prompt_style=None, split_token="%SPLIT%"):
    """
    Tokenize the prompt using the provided tokenizer and prompt style.

    Args:
        prompt (str): The input prompt to tokenize.
        prompt_style: The style of the prompt (e.g., "graph", "text").
        tokenizer: The tokenizer to use for tokenization.

    Returns:
        list: A list of tokenized inputs.
    """
    nodes = prompt.split(split_token)
    encoded_prompt = []
    for node in nodes:
        # Tokenize each node and add to the encoded prompt
        tokenized_node = tokenizer.encode(node + " ")
        encoded_prompt.extend(tokenized_node)
    encoded_prompt = torch.tensor(encoded_prompt)    
    if prompt_style is None:
        return encoded_prompt
    encoded_starting_token = tokenizer.encode(
        prompt_style.starting_token()
    )
    encoded_ending_token = tokenizer.encode(
        prompt_style.ending_token()
    )
    encoded_prompt = torch.cat(
        (
            encoded_starting_token,
            encoded_prompt,
            encoded_ending_token,
        )
    )
    return encoded_prompt
    