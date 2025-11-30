import os
import dataclasses
import numpy as np
import numpy.typing as npt
import matplotlib.patches
import matplotlib.figure
import matplotlib.axes
import matplotlib.pyplot as plt


def set_matplotlib_rc_params():
    plt.rcParams.update(
        {
            "font.size": 14,
            "font.family": "serif",
            "font.serif": "EB Garamond",
            "mathtext.fontset": "cm",
            "text.usetex": False,
            "lines.linewidth": 1.5,
        }
    )


def pages_dir() -> str:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(this_dir)
    return os.path.join(root_dir, "pages")


set_matplotlib_rc_params()


@dataclasses.dataclass
class Node:
    coords: tuple[float, float]
    text: str
    size: float


def plot_node(axes, node: Node):
    axes.add_patch(
        matplotlib.patches.Circle(
            node.coords, node.size, facecolor="none", edgecolor="k"
        )
    )
    axes.text(*node.coords, node.text, ha="center", va="center")


def plot_edge(axes, node1: Node, node2: Node):
    axes.add_patch(
        matplotlib.patches.FancyArrowPatch(
            node1.coords,
            node2.coords,
            arrowstyle="->",
            mutation_scale=20,
            shrinkA=node1.size,
            shrinkB=node2.size,
        )
    )


def make_graph_figure(
    viewport_pts: npt.ArrayLike,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    viewport_pts = np.array(viewport_pts)
    figure, axes = plt.subplots(
        1,
        1,
        figsize=(viewport_pts[1] - viewport_pts[0]) / 72,
        dpi=300,
        gridspec_kw={
            "left": 0,
            "right": 1,
            "top": 1,
            "bottom": 0,
        },
    )
    axes.set_xlim(*viewport_pts[:, 0])
    axes.set_ylim(*viewport_pts[:, 1])
    axes.set_xticks([])
    axes.set_yticks([])
    axes.axis("off")
    return figure, axes
