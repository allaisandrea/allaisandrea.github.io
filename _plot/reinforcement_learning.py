from typing import Callable, Sequence
import sys
import abc
import os
import dataclasses
import matplotlib.collections
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib
from matplotlib.path import Path
import lib


def plot_graphical_model(output_path: str):
    spacing_x = 80

    r_nodes = [
        lib.Node((spacing_x * (i + 0.5), 100), text, 17)
        for i, text in enumerate([r"$\mathbf{r}_0$", r"$\mathbf{r}_1$", r"$\mathbf{r}_2$",  ""])
    ]
    s_nodes = [
        lib.Node((spacing_x * i, 50), text, 17)
        for i, text in enumerate([r"$\mathbf{s}_0$", r"$\mathbf{s}_1$", r"$\mathbf{s}_2$", r"$\mathbf{s}_3$", ""])
    ]
    a_nodes = [
        lib.Node((spacing_x * (i + 0.5), 0), text, 17)
        for i, text in enumerate([r"$\mathbf{a}_0$", r"$\mathbf{a}_1$", r"$\mathbf{a}_2$", ""])
    ]

    figure, axes = lib.make_graph_figure(np.array([[-20, -20], [310, 120]]))
    for node in s_nodes[:-1] + a_nodes[:-1] + r_nodes[:-1]:
        lib.plot_node(axes, node)
    for node1, node2 in zip(s_nodes[:-1], s_nodes[1:]):
        lib.plot_edge(axes, node1, node2)
    for node1, node2 in zip(s_nodes[:-1], a_nodes):
        lib.plot_edge(axes, node1, node2)
    for node1, node2 in zip(s_nodes[:-1], r_nodes):
        lib.plot_edge(axes, node1, node2)
    for node1, node2 in zip(a_nodes[:-1], s_nodes[1:-1]):
        lib.plot_edge(axes, node1, node2)
    for node1, node2 in zip(a_nodes[:-1], r_nodes[:-1]):
        lib.plot_edge(axes, node1, node2)
    figure.savefig(os.path.join(output_path, "graphical_model.svg"))


if __name__ == "__main__":
    task = sys.argv[1]
    output_path = os.path.join(lib.pages_dir(), "reinforcement_learning")
    match task:
        case "graphical_model":
            plot_graphical_model(output_path)
        case _:
            raise ValueError(f"Unknown task: {task}")
