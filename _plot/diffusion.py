from typing import Callable, Sequence
import os
import functools
import dataclasses
import matplotlib.collections
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib
import lib


def bin_centers(x0, x1, n):
    return x0 + (np.arange(n) + 0.5) / n * (x1 - x0)


def normal_log_p_and_score(x, mu, sigma):
    log_p = (
        -np.sum(np.square((x - mu) / sigma) / 2, axis=-1)
        - np.log(2 * np.pi * np.square(sigma)) / 2
    )
    score = -(x - mu) / np.square(sigma)
    return log_p, score


def combine_log_p_and_score(pi1, log_p1, score1, pi2, log_p2, score2):
    z1 = log_p1 + np.log(pi1 / (pi1 + pi2))
    z2 = log_p2 + np.log(pi2 / (pi1 + pi2))
    w1 = np.exp(-np.maximum(z2 - z1, 0))
    w2 = np.exp(-np.maximum(z1 - z2, 0))
    log_p = np.maximum(z1, z2) + np.log(w1 + w2)
    score = (w1[..., None] * score1 + w2[..., None] * score2) / (
        w1[..., None] + w2[..., None]
    )
    return log_p, score


def make_conditional_log_p_function(x0, sigma0, alpha, sigma):
    def compute_log_p(t, x):
        alpha_t = alpha(t)
        sigma_t = sigma(t)
        var = (alpha_t * sigma0) ** 2 + sigma_t**2
        log_p = -np.square(x - alpha_t * x0) / (2 * var) - np.log(2 * np.pi * var) / 2
        d_log_p = -(x - alpha_t * x0) / var
        return log_p, d_log_p

    return compute_log_p


def combine_log_p_functions(pi1, log_p1, pi2, log_p2):
    def compute_log_p(t, x):
        z1, d_log_p1 = log_p1(t, x)
        z2, d_log_p2 = log_p2(t, x)
        z1 = z1 + np.log(pi1 / (pi1 + pi2))
        z2 = z2 + np.log(pi2 / (pi1 + pi2))
        w1 = np.exp(-np.maximum(z2 - z1, 0))
        w2 = np.exp(-np.maximum(z1 - z2, 0))
        log_p = np.maximum(z1, z2) + np.log(w1 + w2)
        d_log_p = (w1 * d_log_p1 + w2 * d_log_p2) / (w1 + w2)
        return log_p, d_log_p

    return compute_log_p


def softmax(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    xmax = np.amax(x, axis=axis)
    exp_x = np.exp(x - xmax)
    return exp_x / np.sum(exp_x, axis=axis)


def sample_x0_given_xt(
    rng: np.random.Generator,
    size: int,
    weights: Sequence[float],
    mu_0: Sequence[float],
    sigma_0: Sequence[float],
    x_t: float,
    alpha_t: float,
    sigma_t: float,
):
    weights = np.array(weights)
    mu_0 = np.array(mu_0)
    sigma_0 = np.array(sigma_0)

    (n_mixture,) = weights.shape
    assert mu_0.shape == (n_mixture,)
    assert sigma_0.shape == (n_mixture,)

    s2 = 1 / (1 / np.square(sigma_0) + np.square(alpha_t / sigma_t))
    mu = s2 * (mu_0 / np.square(sigma_0) + alpha_t * x_t / np.square(sigma_t))
    c = s2 * np.square((x_t - alpha_t * mu_0) / (sigma_0 * sigma_t))
    s = np.sqrt(s2)

    weights = weights * softmax(-c / 2) / s
    weights_cumsum = np.cumsum(weights)
    index = np.sum(
        np.greater(
            rng.uniform(0, weights_cumsum[-1], size)[:, None], weights_cumsum[None]
        ),
        axis=1,
    )
    mu = mu[index]
    s = s[index]
    return rng.normal(mu, s)


def make_sde_params(alpha, d_alpha, sigma, d_sigma):
    def lambda_(t):
        return -d_alpha(t) / alpha(t)

    def gsq(t):
        return 2 * sigma(t) ** 2 * (d_sigma(t) / sigma(t) - d_alpha(t) / alpha(t))

    return lambda_, gsq


def integrate_flow(x_1, lambda_, gsq, compute_log_p, nt):
    dt = 1 / nt
    t_list = [1]
    x_list = [x_1.copy()]
    for i in range(nt, 0, -1):
        t_i = (i - 0.5) / nt
        _, d_log_p = compute_log_p(t_i, x_list[-1])
        x_list.append(
            x_list[-1] * (1 + lambda_(t_i) * dt) + gsq(t_i) * d_log_p / 2 * dt
        )
        t_list.append((i - 1) / nt)
    return np.array(t_list), np.array(x_list)


def sample_diffusion(x_0, lambda_, gsq, nt, rng):
    z = rng.normal(size=(nt, len(x_0)))
    dt = 1 / nt
    t_list = [0]
    x_list = [x_0]
    for i in range(nt):
        t = t_list[-1] + 0.5 * dt
        x_list.append(
            x_list[-1] * (1 - lambda_(t) * dt) + np.sqrt(gsq(t)) * np.sqrt(dt) * z[i]
        )
        t_list.append((i + 1) / nt)
    return np.array(t_list), np.array(x_list)


def plot_flow(
    compute_log_p, x_range, n_mesh_t=1024, n_mesh_x=1024, include_pdf=True
) -> tuple[matplotlib.figure.Figure, list[matplotlib.axes.Axes]]:
    t_range = [0, 1]
    t_mesh = bin_centers(*t_range, n_mesh_t)
    x_mesh = bin_centers(*x_range, n_mesh_x)
    t_mesh = t_mesh[None, :]
    x_mesh = x_mesh[:, None]
    log_p, _ = compute_log_p(t_mesh, x_mesh)
    log_p0, _ = compute_log_p(0, x_mesh)
    log_p1, _ = compute_log_p(1, x_mesh)
    figure = plt.figure(figsize=(7, 5), dpi=300)
    grid_spec = matplotlib.gridspec.GridSpec(
        nrows=2,
        ncols=3,
        wspace=0.08,
        hspace=0.08,
        left=0.09,
        right=0.98,
        top=0.98,
        bottom=0.14,
        width_ratios=[0.1, 0.8, 0.1],
        height_ratios=[0.2, 0.8],
    )
    axes = [None] * 4
    axes[0] = figure.add_subplot(grid_spec[1, 0])
    axes[1] = figure.add_subplot(grid_spec[1, 1], sharey=axes[0])
    axes[2] = figure.add_subplot(grid_spec[1, 2], sharey=axes[0])
    axes[3] = figure.add_subplot(grid_spec[0, 1], sharex=axes[1])

    axes[0].plot(np.exp(log_p0), x_mesh, c="black")
    axes[2].plot(np.exp(log_p1), x_mesh, c="black")
    axes[0].set_xlim(0, None)
    axes[2].set_xlim(0, None)
    axes[0].set_xlabel("$p(x, 0)$")
    axes[2].set_xlabel("$p(x, 1)$")
    axes[0].set_xticks([])
    axes[1].yaxis.set_tick_params(labelleft=False)
    axes[2].yaxis.set_tick_params(labelleft=False)
    axes[3].xaxis.set_tick_params(labelleft=False)
    axes[2].set_xticks([])
    axes[0].set_ylabel("$x$")
    axes[0].grid()
    axes[2].grid()

    levels = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0, 2.0, 1000]) / 2.5
    if include_pdf:
        axes[1].contourf(
            t_mesh[0, :], x_mesh[:, 0], np.exp(log_p), levels=levels, vmax=2 / 2.5
        )
    axes[1].set_xlim(*t_range)
    axes[1].set_ylim(*x_range)
    axes[1].set_aspect("auto")
    axes[1].set_xlabel("$t$")
    figure.align_xlabels(axes)
    return figure, axes


def plot_schedule(axes, alpha, sigma):
    t_list = np.linspace(0, 1, 128)
    axes.plot(t_list, alpha(t_list), c="k")
    axes.plot(t_list, sigma(t_list), c="k", linestyle="--")
    axes.annotate(
        r"$\alpha_t$", (0.1, alpha(0.1)), (0, -15), textcoords="offset points"
    )
    axes.annotate(
        r"$\sigma_t$", (0.9, sigma(0.9)), (0, -15), textcoords="offset points"
    )
    axes.grid()
    axes.set_ylim(0, 1.10)


def plot_diffusion(output_path: str):
    pi1 = 0.8
    x1 = -1.0
    sigma1 = 0.1
    pi2 = 0.2
    x2 = +2.0
    sigma2 = 0.1
    # alpha = lambda t: 1 - t
    # d_alpha = lambda t: -1
    # sigma = lambda t: t
    # d_sigma = lambda t: 1
    alpha = lambda t: np.cos(np.pi * t / 2)
    d_alpha = lambda t: -np.pi / 2 * np.sin(np.pi * t / 2)
    sigma = lambda t: np.sin(np.pi * t / 2)
    d_sigma = lambda t: np.pi / 2 * np.cos(np.pi * t / 2)
    lambda_, gsq = make_sde_params(alpha, d_alpha, sigma, d_sigma)

    compute_log_p = combine_log_p_functions(
        pi1,
        make_conditional_log_p_function(x0=x1, sigma0=sigma1, alpha=alpha, sigma=sigma),
        pi2,
        make_conditional_log_p_function(x0=x2, sigma0=sigma2, alpha=alpha, sigma=sigma),
    )
    figure, axes = plot_flow(compute_log_p, x_range=(-3, 3))
    plot_schedule(axes[3], alpha, sigma)
    rng = np.random.default_rng(0)
    n_sample = 16
    x_0 = np.where(
        rng.uniform(size=n_sample) < pi1,
        rng.normal(x1, sigma1, size=n_sample),
        rng.normal(x2, sigma2, size=n_sample),
    )
    t_list, x_list = sample_diffusion(x_0, lambda_=lambda_, gsq=gsq, nt=8192, rng=rng)
    axes[1].plot(t_list, x_list, c="white", linewidth=0.5, alpha=0.5)
    figure.savefig(os.path.join(output_path, "diffusion.png"))

    figure, axes = plot_flow(compute_log_p, x_range=(-3, 3))
    plot_schedule(axes[3], alpha, sigma)
    t_list, x_list = integrate_flow(
        x_1=scipy.stats.norm.ppf(bin_centers(0, 1.0, 16)),
        lambda_=lambda_,
        gsq=gsq,
        compute_log_p=compute_log_p,
        nt=2048,
    )
    axes[1].plot(t_list, x_list, c="white", linewidth=1, alpha=0.5)
    figure.savefig(os.path.join(output_path, "probability_flow.png"))


def plot_p_and_score(output_path):
    n_x0 = 384
    n_x1 = 256
    x0_mesh = bin_centers(-3, 3, n_x0)
    x1_mesh = bin_centers(-5, 5, n_x1)
    x0_mesh = np.broadcast_to(x0_mesh[:, None], (n_x0, n_x1))
    x1_mesh = np.broadcast_to(x1_mesh[None, :], (n_x0, n_x1))
    x_mesh = np.stack([x1_mesh, x0_mesh], axis=2)
    log_p, score = combine_log_p_and_score(
        0.4,
        *normal_log_p_and_score(x_mesh, [-3, -1], 2),
        0.6,
        *normal_log_p_and_score(x_mesh, [1, 2], 2.5),
    )
    figure, axes = plt.subplots(
        1,
        1,
        figsize=(5, 3.5),
        dpi=300,
    )
    axes.contourf(x1_mesh[0, :], x0_mesh[:, 0], np.exp(log_p))
    axes.streamplot(
        x1_mesh[0, :], x0_mesh[:, 0], score[:, :, 0], score[:, :, 1], color="white"
    )
    axes.set_aspect("equal")
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(r"$p$ and $\nabla \log p$", pad=12)
    figure.tight_layout()
    figure.savefig(os.path.join(output_path, "score_function.png"))


def plot_slope_distribution(
    t_point: float, output_tag: str | None, output_path: str | None
):
    pi1 = 0.8
    x1 = -1.0
    sigma1 = 0.1
    pi2 = 0.2
    x2 = +2.0
    sigma2 = 0.1
    alpha = lambda t: 1 - t
    d_alpha = lambda t: -1
    sigma = lambda t: t
    d_sigma = lambda t: 1
    lambda_, gsq = make_sde_params(alpha, d_alpha, sigma, d_sigma)

    compute_log_p = combine_log_p_functions(
        pi1,
        make_conditional_log_p_function(x0=x1, sigma0=sigma1, alpha=alpha, sigma=sigma),
        pi2,
        make_conditional_log_p_function(x0=x2, sigma0=sigma2, alpha=alpha, sigma=sigma),
    )

    nt = 2048
    t_list, x_list = integrate_flow(
        x_1=scipy.stats.norm.ppf(bin_centers(0, 1.0, 16)),
        lambda_=lambda_,
        gsq=gsq,
        compute_log_p=compute_log_p,
        nt=nt,
    )

    i_point, j_point = int(nt * (1 - t_point)), 12
    t_point = t_list[i_point]
    x_t_point = x_list[i_point, j_point]
    alpha_t_point = alpha(t_point)
    sigma_t_point = sigma(t_point)
    rng = np.random.default_rng(0)
    x0_line = sample_x0_given_xt(
        rng=rng,
        size=50,
        weights=[pi1, pi2],
        mu_0=[x1, x2],
        sigma_0=[sigma1, sigma2],
        x_t=x_t_point,
        alpha_t=alpha_t_point,
        sigma_t=sigma_t_point,
    )
    x1_line = (x_t_point - alpha_t_point * x0_line) / sigma_t_point
    lines = np.reshape(
        np.stack(
            [np.zeros_like(x0_line), x0_line, np.ones_like(x1_line), x1_line], axis=-1
        ),
        (-1, 2, 2),
    )
    figure, axes = plot_flow(compute_log_p, x_range=(-3, 3), include_pdf=False)
    plot_schedule(axes[3], alpha, sigma)
    axes[1].plot(t_list, x_list, c="lightgray", linewidth=1.0)
    axes[1].add_artist(
        matplotlib.collections.LineCollection(
            lines, linewidth=1.0, color="orangered", alpha=0.5
        )
    )
    axes[1].plot(
        [t_point],
        [x_t_point],
        marker="o",
        markersize=3,
        markerfacecolor="k",
        markeredgecolor="k",
    )
    axes[1].annotate(
        "$(t, x_t)$",
        (t_point, x_t_point),
        (0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
    )
    axes[1].plot(t_list, x_list[:, j_point], c="black", linewidth=1.5)
    if output_path is not None:
        assert output_tag is not None
        figure.savefig(
            os.path.join(output_path, f"slope_distribution_{output_tag}.svg"),
            format="SVG",
        )
    return figure


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


def make_graph_figure(viewport_pts):
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


def plot_graphical_models(output_path):
    spacing_x = 80
    x_text = [
        r"$x_0$",
        r"$x_{\Delta t}$",
        r"$x_{2\Delta t}$",
        r"$x_{3\Delta t}$",
        r"$x_{4\Delta t}$",
    ]
    z_text = [
        r"$z_{\Delta t}$",
        r"$z_{2\Delta t}$",
        r"$z_{3\Delta t}$",
    ]
    zbar_text = [
        r"$\bar{z}_{\Delta t}$",
        r"$\bar{z}_{2\Delta t}$",
        r"$\bar{z}_{3\Delta t}$",
    ]
    x_nodes = [Node((spacing_x * i, 0), text, 17) for i, text in enumerate(x_text)]
    z_nodes = [
        Node((spacing_x * (i + 1), 80), text, 17) for i, text in enumerate(z_text)
    ]
    zbar_nodes = [
        Node((spacing_x * (i + 1), 80), text, 17) for i, text in enumerate(zbar_text)
    ]

    figure, axes = make_graph_figure(np.array([[-20, -20], [310, 100]]))
    for node in x_nodes[:-1] + z_nodes:
        plot_node(axes, node)
    for node1, node2 in zip(x_nodes[:-1], x_nodes[1:]):
        plot_edge(axes, node1, node2)
    for node1, node2 in zip(z_nodes, x_nodes[1:]):
        plot_edge(axes, node1, node2)
    figure.savefig(os.path.join(output_path, "forward_process.png"))

    figure, axes = make_graph_figure(np.array([[-20, -20], [310, 100]]))
    for node in x_nodes[:-1] + zbar_nodes:
        plot_node(axes, node)
    for node2, node1 in zip(x_nodes[:-1], x_nodes[1:]):
        plot_edge(axes, node1, node2)
    for node1, node2 in zip(zbar_nodes, x_nodes[:-1]):
        plot_edge(axes, node1, node2)
    figure.savefig(os.path.join(output_path, "backward_process.png"))


if __name__ == "__main__":
    output_path = os.path.join(lib.pages_dir(), "diffusion")
    plot_diffusion(output_path)
    plot_p_and_score(output_path)
    plot_graphical_models(output_path)
    for i, t_point in enumerate([0.1, 0.4, 0.6, 0.9]):
        plot_slope_distribution(
            t_point=t_point, output_tag=str(i), output_path=output_path
        )
