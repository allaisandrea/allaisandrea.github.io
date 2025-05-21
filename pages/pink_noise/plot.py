from typing import Callable
import dataclasses
import numpy as np
import matplotlib.pyplot as plt
import scipy
import matplotlib


def compute_correlation(
    n: int, alpha: float, use_sin: bool = False, uv_norm: bool = False
):
    assert n % 2 == 0
    k = np.arange(n // 2 + 1)
    with np.errstate(divide="ignore"):
        if use_sin:
            kernel_k = (
                np.where(k > 0, np.power(n / np.pi * np.sin(np.pi * k / n), -alpha), 0)
                / 2
            )
        else:
            kernel_k = np.where(k > 0, np.power(np.minimum(k, n - k), -alpha), 0) / 2
    if uv_norm:
        if alpha < 1:
            kernel_k *= np.pow(n / (2 * np.pi), alpha - 1) / (
                np.sin(np.pi * alpha / 2) * scipy.special.gamma(1 - alpha)
            )
        else:
            kernel_k /= scipy.special.zeta(alpha)
    return np.fft.irfft(kernel_k, norm="forward")


def plot_n_x_gg_1():
    n = 1 << 15
    alpha_list = [0.5, 2.0]
    correlation_list = [compute_correlation(n, alpha) for alpha in alpha_list]
    figure, axes = plt.subplots(
        1,
        1,
        figsize=(6, 4),
        squeeze=False,
        gridspec_kw=dict(top=0.85, bottom=0.15, left=0.12, right=0.97),
    )
    axes = axes.flatten()
    y_min = -(1 - np.power(2, 1 - alpha_list[1])) * scipy.special.zeta(alpha_list[1])
    y_max = scipy.special.zeta(alpha_list[1])
    axes[0].plot(
        np.arange(n) / n, correlation_list[0], color="k", label=r"$\alpha < 1$"
    )
    axes[0].plot(
        np.arange(n) / n,
        correlation_list[1],
        color="k",
        linestyle="--",
        label=r"$\alpha \geq 1$",
    )
    axes[0].set_ylim(-1.25, 3)
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel(r"$\hat{x}$")
    axes[0].set_xticks([0, 0.5, 1])
    axes[0].set_xticklabels(["0", "$1 / 2$", "$1$"])
    axes[0].set_yticks([0, y_min, y_max])
    axes[0].set_yticklabels(["0", r"-$\eta(\alpha)$", r"$\zeta(\alpha)$"])
    axes[0].grid()
    axes[0].legend(framealpha=1)
    figure.suptitle(r"$\lim_{N\to\infty}\mathrm{E}[\psi_0\psi_{N\hat{x}}]$")
    figure.savefig("n_x_gg_1.svg")
    plt.close(figure)


def plot_n_gg_x_1():
    n = 1 << 25
    alpha_list = [0.25, 0.5, 0.75]
    correlation_list = [
        compute_correlation(n, alpha, use_sin=True, uv_norm=True)
        for alpha in alpha_list
    ] + [np.ones(n)]
    labels = [f"$\\alpha = {alpha}$" for alpha in alpha_list] + [f"$\\alpha \\geq 1$"]
    linestyles = ["solid", "dashed", "dashdot", "dotted"]

    n_take_max = 100
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(6, 5),
        sharey=True,
        squeeze=False,
        gridspec_kw=dict(top=0.85, bottom=0.15, left=0.12, right=0.97, wspace=0.12),
    )
    axes = axes.flatten()
    x = np.arange(n_take_max)
    for i, correlation in enumerate(correlation_list):
        n = len(correlation)
        n_take = min(n_take_max, n)
        axes[0].plot(
            x[:n_take],
            correlation[:n_take],
            color="k",
            label=labels[i],
            linestyle=linestyles[i],
        )
        axes[0].plot(
            x[:11],
            correlation[:11],
            color="k",
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markersize=4,
        )
        axes[1].plot(
            x[1:n_take],
            correlation[1:n_take],
            color="k",
            linestyle=linestyles[i],
        )
        axes[1].plot(
            x[1:11],
            correlation[1:11],
            color="k",
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markersize=4,
        )

    axes[0].legend(framealpha=1)
    axes[0].grid()
    axes[1].grid()
    axes[0].set_xlabel("$x$")
    axes[1].set_xlabel("$x$")
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xscale("log")
    axes[0].set_xlim(0, 20)
    axes[1].set_xlim(1, n_take_max)
    axes[0].set_ylim(0.01, 10)
    axes[1].set_ylim(0.01, 10)
    figure.suptitle(r"$\lim_{N\to\infty}\mathrm{E}[\psi_0\psi_x]$")
    figure.savefig("n_gg_x_1.svg")
    plt.close(figure)


plt.style.use("../../_plot/mplstyle.rc")

plot_n_x_gg_1()
plot_n_gg_x_1()
