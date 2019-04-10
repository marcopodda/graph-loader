import io
import os
import pickle
from pathlib import Path
import requests
import yaml
import zipfile

import networkx as nx
import numpy as np
import sklearn

# useful directories
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "DATA"
REPO_DIR = DATA_DIR / "REPO"
RAW_DIR = DATA_DIR / "RAW"
CFG_DIR = ROOT_DIR / "CONFIG"

DEFAULT_ATTRS = dict(
    undirected=True,
    has_graphlabels=False,
    has_nodeattrs=False,
    has_edgeattrs=False
)


def get_config(name, config_filename):
    if config_filename is None:
        config_filename = f"{name}.yaml"

    config_path = os.path.join(CFG_DIR, config_filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError("""
            Error loading config file. You must either:
            1) provide a .yaml config file;
            2) use another dataset. """)

    with open(config_path, "r") as stream:
        config = yaml.load(stream)

    return config


def fetch_data(name, url):
    if not os.path.exists(RAW_DIR):
        os.makedirs(RAW_DIR)

    response = requests.get(url)
    stream = io.BytesIO(response.content)
    with zipfile.ZipFile(stream) as z:
        for fname in z.namelist():
            z.extract(fname, RAW_DIR)


def parse_dataset(name, path, undirected=True, has_graphlabels=True,
                  has_edgelabels=True, has_nodelabels=True,
                  has_nodeattrs=True, has_edgeattrs=True, sep=","):

    # setup paths
    path = Path(path)
    indicator_path = path / f'{name}_graph_indicator.txt'
    edges_path = path / f'{name}_A.txt'
    graph_labels_path = path / f'{name}_graph_labels.txt'
    node_labels_path = path / f'{name}_node_labels.txt'
    edge_labels_path = path / f'{name}_edge_labels.txt'
    node_attrs_path = path / f'{name}_node_attributes.txt'
    edge_attrs_path = path / f'{name}_edge_attributes.txt'

    # setup data structures
    node_labels = dict()
    edge_labels = dict()
    graph_labels = dict()
    node_attrs = dict()
    edge_attrs = dict()
    node_to_graph = dict()
    edge_to_line = dict()

    # get graph ids
    with open(indicator_path, "r") as f:
        for (i, line) in enumerate(f, 1):
            graph_id = int(line[:-1])
            node_to_graph[i] = graph_id
            graph_labels[graph_id] = dict()
            if graph_id not in node_labels:
                node_labels[graph_id] = dict()
            if graph_id not in edge_labels:
                edge_labels[graph_id] = dict()
            if graph_id not in node_attrs:
                node_attrs[graph_id] = dict()
            if graph_id not in edge_attrs:
                edge_attrs[graph_id] = dict()

    num_graphs = len(graph_labels.keys())

    # get edges info
    with open(edges_path, "r") as f:
        for (i, line) in enumerate(f, 1):
            line = line[:-1].replace(" ", "").split(sep)
            edge1, edge2 = [int(e) for e in line]
            edge_to_line[i] = (edge1, edge2)

    # (optionally) get graph labels
    if has_graphlabels:
        with open(graph_labels_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                graph_label = {"label": int(line[:-1])}
                graph_labels[i] = graph_label

    # get node labels
    if has_nodelabels:
        with open(node_labels_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                node_label = {"label": int(line[:-1])}
                node_labels[node_to_graph[i]][i] = node_label

    # (optionally) get node attributes
    if node_attrs_path.exists():
        with open(node_attrs_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                line = line[:-1].replace(" ", "").split(sep)
                attrs = np.array([float(a) for a in line])
                node_attrs[node_to_graph[i]][i] = attrs

    # (optionally) get edge labels
    if has_edgelabels and edge_labels_path.exists():
        with open(edge_labels_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                edge = edge_to_line[i]
                edge_label = {"label": int(line[:-1])}
                edge_labels[node_to_graph[edge[0]]][edge] = edge_label
                if undirected:
                    edges = (edge[1], edge[0])
                    edge_labels[node_to_graph[edge[1]]][edges] = edge_label

    # (optionally) get edge attributes
    if edge_attrs_path.exists():
        with open(edge_attrs_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                edge = edge_to_line[i]
                line = line[:-1].replace(" ", "").split(sep)
                attrs = np.array([float(a) for a in line])
                edge_attrs[node_to_graph[edge[0]]][edge] = attrs
                if undirected:
                    edges = (edge[1], edge[0])
                    edge_attrs[node_to_graph[edge[1]]][edges] = attrs

    return num_graphs, node_labels, edge_labels, graph_labels, node_attrs, edge_attrs


def build_graph(index, node_labels, edge_labels, graph_labels, node_attrs, edge_attrs):
    G, target = nx.Graph(), None

    if graph_labels[index]:
        [(label_name, target)] = graph_labels[index].items()
        G.graph[label_name] = target
        G.graph["has_target"] = True

    # add nodes
    nodes = [(k, v) for (k, v) in node_labels[index].items()]
    G.add_nodes_from(nodes)

    # add edges
    edges = [(k1, k2, v) for ((k1, k2), v) in edge_labels[index].items()]
    G.add_edges_from(edges)

    # (optionally) add node attributes
    if node_attrs:
        node_attrs = node_attrs[index]
        for (node_idx, attrs) in node_attrs.items():
            G.node[node_idx]["attrs"] = node_attrs[node_idx]

    # (optionally) add edge attributes
    if edge_attrs:
        edge_attrs = edge_attrs[index]
        for ((e1, e2), attrs) in edge_attrs.items():
            G[e1][e2]["attrs"] = edge_attrs[(e1, e2)]

    return G, target


def build_graphs(N, *data, add_targets=True):
    graphs, targets = [], []
    for i in range(1, N + 1):
        G, target = build_graph(i, *data)
        graphs.append(G)

        if add_targets:
            targets.append(target)
    return graphs, targets


def load_data(name, config_filename=None):
    # if loading fails, try to build the dataset from text files
    try:
        dataset_path = REPO_DIR / f"{name}.pkl"
        dataset = pickle.load(open(dataset_path, "rb"))
    except Exception:
        config = get_config(name, config_filename)
        dataset_attrs = DEFAULT_ATTRS.copy()
        dataset_attrs.update(config["attributes"])

        dataset_path = RAW_DIR / name
        if not dataset_path.exists():
            fetch_data(name, config["url"])

        graphs, targets = [], []
        N, *data = parse_dataset(name, dataset_path, **dataset_attrs)
        add_targets = config["attributes"]["has_graphlabels"]
        graphs, targets = build_graphs(N, *data, add_targets=add_targets)

        dataset = sklearn.utils.Bunch(graphs=graphs)
        if add_targets:
            dataset["targets"] = targets

        if not REPO_DIR.exists():
            os.makedirs(REPO_DIR)

        dataset_path = REPO_DIR / f"{name}.pkl"
        pickle.dump(dataset, open(dataset_path, "wb"))

    return dataset


def list_datasets():
    names = []

    for filename in CFG_DIR.glob("[!\.]*.yaml"):
        names.append(filename.stem)

    return sorted(names)
