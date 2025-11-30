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
        for i, text in enumerate(
            [r"$\mathbf{r}_0$", r"$\mathbf{r}_1$", r"$\mathbf{r}_2$", ""]
        )
    ]
    s_nodes = [
        lib.Node((spacing_x * i, 50), text, 17)
        for i, text in enumerate(
            [
                r"$\mathbf{s}_0$",
                r"$\mathbf{s}_1$",
                r"$\mathbf{s}_2$",
                r"$\mathbf{s}_3$",
                "",
            ]
        )
    ]
    a_nodes = [
        lib.Node((spacing_x * (i + 0.5), 0), text, 17)
        for i, text in enumerate(
            [r"$\mathbf{a}_0$", r"$\mathbf{a}_1$", r"$\mathbf{a}_2$", ""]
        )
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


def plot_trpo_diagram(output_path: str):
    theta = np.linspace(-0.5, 0.7, 128)
    c0 = 0.0
    c1 = 0.0
    c20 = -1.0
    c21 = -0.4
    c22 = -2.0
    theta_0 = -0.35

    def f0(theta):
        return c0 + c1 * theta + c20 * theta**2

    def df0(theta):
        return c1 + 2 * c20 * theta

    def f1(theta):
        d_theta = theta - theta_0
        return f0(theta_0) + df0(theta_0) * d_theta + c21 * d_theta**2

    def f2(theta):
        d_theta = theta - theta_0
        return f0(theta_0) + df0(theta_0) * d_theta + c22 * d_theta**2

    theta_max_0 = -c1 / (2 * c20)
    theta_max_1 = theta_0 - df0(theta_0) / (2 * c21)
    theta_max_2 = theta_0 - df0(theta_0) / (2 * c22)
    figure, axes = plt.subplots(
        1,
        1,
        figsize=(5, 3),
        dpi=300,
        gridspec_kw=dict(top=0.99, bottom=0.15, left=0.01, right=0.99),
    )
    axes.plot(theta, f0(theta), color="black")
    axes.plot(theta, f1(theta), color="black")
    axes.plot(theta, f2(theta), color="black")
    axes.plot(
        [theta_0, theta_max_0, theta_max_1, theta_max_1, theta_max_2, theta_max_2],
        [
            f0(theta_0),
            f0(theta_max_0),
            f1(theta_max_1),
            f0(theta_max_1),
            f2(theta_max_2),
            f0(theta_max_2),
        ],
        color="black",
        marker="o",
        linestyle="none",
        ms=4,
    )
    axes.axvline(theta_0, color="black", linewidth=1)
    axes.axvline(theta_max_0, color="black", linewidth=1)
    axes.axvline(theta_max_1, color="black", linewidth=1)
    axes.axvline(theta_max_2, color="black", linewidth=1)
    axes.annotate(
        r"$\eta(\theta)$",
        (0.35, f0(0.35)),
        (5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
    )
    axes.annotate(
        r"$J_{\mathrm{naive}}(\theta, \theta_0)$",
        (0.22, f1(0.22)),
        (0, -5),
        textcoords="offset points",
        ha="left",
        va="top",
    )
    axes.annotate(
        r"$J_{\mathrm{TRPO}}(\theta, \theta_0)$",
        (0.22, f2(0.22)),
        (5, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
    )
    axes.set_xticks([theta_0, theta_max_0, theta_max_1, theta_max_2])
    axes.set_xticklabels(
        [
            r"$\theta_0$",
            r"$\theta_\mathrm{max}$",
            r"$\theta_\mathrm{naive}$",
            r"$\theta_\mathrm{TRPO}$",
        ]
    )
    axes.set_yticks([])
    axes.set_ylim(-0.4, 0.25)
    axes.set_xlabel(r"$\theta$")
    axes.xaxis.set_label_coords(1.0, -0.03)
    figure.savefig(os.path.join(output_path, "trpo_diagram.svg"))


if __name__ == "__main__":
    task = sys.argv[1]
    output_path = os.path.join(lib.pages_dir(), "reinforcement_learning")
    match task:
        case "graphical_model":
            plot_graphical_model(output_path)
        case "trpo_diagram":
            plot_trpo_diagram(output_path)
        case _:
            raise ValueError(f"Unknown task: {task}")
