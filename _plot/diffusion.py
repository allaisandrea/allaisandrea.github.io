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


def compute_bin_centers(x0: float, x1: float, n: int) -> np.ndarray:
    return x0 + (np.arange(n) + 0.5) / n * (x1 - x0)


def softmax(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    xmax = np.amax(x, axis=axis)
    exp_x = np.exp(x - xmax)
    return exp_x / np.sum(exp_x, axis=axis)


def compute_normal_log_p_and_score(x: np.ndarray, mean: np.ndarray, var: np.ndarray):
    """Compute log p and score of a normal distribution.
    Args:
        x: [*ldims]
        mean: [*ldims]
        var: [*ldims]
    Returns:
        log_p: [*ldims]
        score: [*ldims]
    """
    log_p = -(np.square((x - mean)) / var + np.log(2 * np.pi) + np.log(var)) / 2
    score = -(x - mean) / var
    return log_p, score


def compute_multivariate_normal_log_p_and_score(
    x: np.ndarray, mean: np.ndarray, var: np.ndarray
):
    """Compute log p and score of a normal distribution.
    Args:
        x: [*ldims, ndim]
        mean: [*ldims, ndim]
        var: [*ldims]
    Returns:
        log_p: [*ldims]
        score: [*ldims, ndim]
    """
    n_dim = x.shape[-1]
    log_p = (
        -(
            np.sum(np.square((x - mean)), axis=-1) / var
            + n_dim * np.log(2 * np.pi)
            + n_dim * np.log(var)
        )
        / 2
    )
    score = -(x - mean) / var[..., None]
    return log_p, score


def compute_mixture_log_p_and_score(log_p: np.ndarray, score: np.ndarray):
    """Compute log p and score of a mixture
    Args:
        log_p: [*ldims, n_mixture]
        score: [*ldims, n_mixture, n_dim]
    Returns:
        log_p: [*ldims]
        score: [*ldims, n_dim]
    """
    max_log_p = np.amax(log_p, axis=-1)
    p = np.exp(log_p - max_log_p[..., None])
    log_p = max_log_p + np.log(np.sum(p, axis=-1))
    score = np.sum(p[..., None] * score, axis=-2) / np.sum(p[..., None], axis=-2)
    return log_p, score


def compute_normal_posterior_and_evidence(
    mean0: np.ndarray, var0: np.ndarray, x: np.ndarray, var: np.ndarray
):
    """Compute the posterior and evidence of a normal distribution.
    Args:
        mu0: [*ldims] prior mean
        sigma0: [*ldims] prior variance
        x: [*ldims] data
        sigma: [*ldims] likelihood variance
    Returns:
        mean1: [*ldims] posterior mean
        var1: [*ldims] posterior variance
        log_evidence: [*ldims] log evidence
    """
    assert var0.shape == mean0.shape
    assert x.shape == mean0.shape
    assert var.shape == mean0.shape
    s2 = var0 + var
    var1 = var0 * var / s2
    mean1 = (mean0 * var + x * var0) / s2
    log_evidence = -(np.square(x - mean0) / s2 + np.log(s2)) / 2
    return mean1, var1, log_evidence


def sample_from_categorical(
    rng: np.random.Generator, weights: np.ndarray, size: int
) -> np.ndarray:
    """Sample from a categorical distribution.
    Args:
        rng: np.random.Generator
        weights: [*ldims, n_mixture]
        size: int
    Returns:
        index: [*ldims, size]
    """
    # [*ldims, n_mixture]
    thresholds = np.cumsum(weights, axis=-1)
    thresholds = thresholds / thresholds[..., -1:]

    # [*ldims, size, 1]
    uniform = rng.uniform(size=(*weights.shape[:-1], size, 1))

    # [*ldims, size]
    index = np.sum(uniform > thresholds[..., None, :], axis=-1)
    return index


class NoiseSchedule(abc.ABC):
    def alpha(self, t: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def d_alpha(self, t: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def sigma(self, t: np.ndarray) -> np.ndarray:
        pass

    @abc.abstractmethod
    def d_sigma(self, t: np.ndarray) -> np.ndarray:
        pass

    def lambda_(self, t: np.ndarray) -> np.ndarray:
        return -self.d_alpha(t) / self.alpha(t)

    def gsq(self, t: np.ndarray) -> np.ndarray:
        return (
            2
            * np.square(self.sigma(t))
            * (self.d_sigma(t) / self.sigma(t) - self.d_alpha(t) / self.alpha(t))
        )

    def sample_diffusion_trajectory(
        self, x_0: np.ndarray, nt: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample the diffusion trajectory of the interpolant.
        Args:
            x_0: [n_particles]
            nt: int
            rng: np.random.Generator
        Returns:
            t_list: [nt + 1]
            x_list: [nt + 1, n_particles]
        """
        (n_particles,) = x_0.shape
        z = rng.normal(size=(nt, n_particles))
        dt = 1 / nt
        t_list = [0.0]
        x_list = [x_0]
        for i in range(nt):
            t = t_list[-1] + 0.5 * dt
            x_list.append(
                x_list[-1] * (1 - self.lambda_(t) * dt)
                + np.sqrt(self.gsq(t)) * np.sqrt(dt) * z[i]
            )
            t_list.append((i + 1) / nt)
        return np.stack(t_list), np.stack(x_list)


class LinearNoiseSchedule(NoiseSchedule):
    def alpha(self, t: np.ndarray) -> np.ndarray:
        return 1 - t

    def d_alpha(self, t: np.ndarray) -> np.ndarray:
        return -1

    def sigma(self, t: np.ndarray) -> np.ndarray:
        return t

    def d_sigma(self, t: np.ndarray) -> np.ndarray:
        return 1


class CosineNoiseSchedule(NoiseSchedule):
    def alpha(self, t: np.ndarray) -> np.ndarray:
        return np.cos(np.pi * t / 2)

    def d_alpha(self, t: np.ndarray) -> np.ndarray:
        return -np.pi / 2 * np.sin(np.pi * t / 2)

    def sigma(self, t: np.ndarray) -> np.ndarray:
        return np.sin(np.pi * t / 2)

    def d_sigma(self, t: np.ndarray) -> np.ndarray:
        return np.pi / 2 * np.cos(np.pi * t / 2)


@dataclasses.dataclass
class GaussianMixture:
    means: np.ndarray
    stddevs: np.ndarray
    weights: np.ndarray

    def __post_init__(self):
        (n_mixture,) = self.weights.shape
        assert self.means.shape == (n_mixture,)
        assert self.stddevs.shape == (n_mixture,)
        self.weights = self.weights / np.sum(self.weights)

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        index = sample_from_categorical(rng, self.weights, size)
        means = self.means[index]
        stddevs = self.stddevs[index]
        return rng.normal(means, stddevs)


@dataclasses.dataclass
class GaussianMixtureInterpolant:
    noise_schedule: NoiseSchedule
    x0_distr: GaussianMixture

    def __post_init__(self):
        (n_mixture,) = self.x0_distr.weights.shape
        assert self.x0_distr.means.shape == (n_mixture,)
        assert self.x0_distr.stddevs.shape == (n_mixture,)
        assert np.all(self.x0_distr.weights >= 0)
        assert np.all(self.x0_distr.stddevs > 0)
        self.x0_distr.weights = self.x0_distr.weights / np.sum(self.x0_distr.weights)

    def compute_log_pdf_and_score(self, t: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Compute log p and score of the interpolant.
        Args:
            t: [*ldims]
            x: [*ldims]
        Returns:
            log_p: [*ldims]
            score: [*ldims]
        """
        # [*ldims, 1]
        t = t[..., None]
        x = x[..., None]
        alpha_t = self.noise_schedule.alpha(t)
        sigma_t = self.noise_schedule.sigma(t)
        # [*ldims, n_mixture]
        mean = alpha_t * self.x0_distr.means
        var = np.square(alpha_t * self.x0_distr.stddevs) + np.square(sigma_t)
        log_p, d_log_p = compute_normal_log_p_and_score(x, mean, var)
        log_p = log_p + np.log(self.x0_distr.weights)
        log_p, d_log_p = compute_mixture_log_p_and_score(log_p, d_log_p[..., None])
        return log_p, np.squeeze(d_log_p, axis=-1)

    def sample_x0_given_xt(
        self,
        rng: np.random.Generator,
        size: int,
        t: np.ndarray,
        x_t: np.ndarray,
    ):
        """Sample x0 given x_t.
        Args:
            rng: np.random.Generator
            size: int
            t: [n_particles]
            x_t: [n_particles]
        Returns:
            x0: [n_particles, size]
        """
        (n_particles,) = t.shape
        assert x_t.shape == (n_particles,)

        # [n_particles]
        alpha_t = self.noise_schedule.alpha(t)
        sigma_t = self.noise_schedule.sigma(t)

        # [n_mixture]
        sigma_0 = self.x0_distr.stddevs
        mu_0 = self.x0_distr.means

        # [n_particles, n_mixture]
        alpha_t, sigma_t, x_t, sigma_0, mu_0 = np.broadcast_arrays(
            alpha_t[:, None], sigma_t[:, None], x_t[:, None], sigma_0, mu_0
        )

        # [n_particles, n_mixture]
        mean1, var1, log_evidence = compute_normal_posterior_and_evidence(
            mean0=alpha_t * mu_0,
            var0=np.square(alpha_t * sigma_0),
            x=x_t,
            var=np.square(sigma_t),
        )
        evidence = np.exp(log_evidence)
        mean1 = mean1 / alpha_t
        stddev1 = np.sqrt(var1) / alpha_t
        weights = self.x0_distr.weights * evidence
        weights = weights / np.sum(weights, axis=-1, keepdims=True)

        # [n_particles, size]
        index = sample_from_categorical(rng, weights, size)

        mean1 = mean1[np.arange(n_particles)[:, None], index]
        stddev1 = stddev1[np.arange(n_particles)[:, None], index]
        return rng.normal(mean1, stddev1)

    def compute_probability_flow_trajectory(
        self, x_1: np.ndarray, nt: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the flow trajectory of the interpolant.
        Args:
            x_1: [*ldims]
            nt: int
        Returns:
            t_list: [nt]
            x_list: [nt, *ldims]
        """
        dt = 1 / nt
        t_list = [1.0]
        x_list = [x_1.copy()]
        for i in range(nt, 0, -1):
            t_i = (i - 0.5) / nt
            _, d_log_p = self.compute_log_pdf_and_score(np.full(1, t_i), x_list[-1])
            x_list.append(
                x_list[-1] * (1 + self.noise_schedule.lambda_(t_i) * dt)
                + self.noise_schedule.gsq(t_i) * d_log_p / 2 * dt
            )
            t_list.append((i - 1) / nt)
        return np.stack(t_list), np.stack(x_list)

    def sample_diffusion_trajectory(self, rng: np.random.Generator, nt: int, nx: int):
        x_0 = self.x0_distr.sample(rng, nx)
        return self.noise_schedule.sample_diffusion_trajectory(x_0, nt, rng)


def plot_schedule(axes: matplotlib.axes.Axes, noise_schedule: NoiseSchedule):
    t_list = np.linspace(0, 1, 128)
    axes.plot(t_list, noise_schedule.alpha(t_list), c="k")
    axes.plot(t_list, noise_schedule.sigma(t_list), c="k", linestyle="--")
    axes.annotate(
        r"$\alpha_t$",
        (0.1, noise_schedule.alpha(0.1)),
        (0, -15),
        textcoords="offset points",
    )
    axes.annotate(
        r"$\sigma_t$",
        (0.9, noise_schedule.sigma(0.9)),
        (0, -15),
        textcoords="offset points",
    )
    axes.grid()
    axes.set_ylim(0, 1.10)


def plot_interpolant_pdf(
    axes: matplotlib.axes.Axes,
    interpolant: GaussianMixtureInterpolant,
    x_range: tuple[float, float],
    n_mesh_t: int = 1024,
    n_mesh_x: int = 1024,
):
    t_bin_centers = compute_bin_centers(0, 1, n_mesh_t)
    x_bin_centers = compute_bin_centers(*x_range, n_mesh_x)
    t_bin_centers = t_bin_centers[None, :]
    x_bin_centers = x_bin_centers[:, None]
    log_p, _ = interpolant.compute_log_pdf_and_score(t_bin_centers, x_bin_centers)
    levels = np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0, 2.0, 1000]) / 2.5
    axes.contourf(
        t_bin_centers[0, :],
        x_bin_centers[:, 0],
        np.exp(log_p),
        levels=levels,
        vmax=2 / 2.5,
    )


@dataclasses.dataclass
class InterpolantPlotTemplate:
    figure: matplotlib.figure.Figure
    schedule_axes: matplotlib.axes.Axes
    data_pdf_axes: matplotlib.axes.Axes
    noise_pdf_axes: matplotlib.axes.Axes
    interpolant_axes: matplotlib.axes.Axes


def make_interpolant_plot_template(
    interpolant: GaussianMixtureInterpolant,
    x_range: tuple[float, float],
    n_mesh_x: int = 1024,
) -> InterpolantPlotTemplate:
    t_range = [0, 1]
    x_bin_centers = compute_bin_centers(*x_range, n_mesh_x)
    log_p0, _ = interpolant.compute_log_pdf_and_score(np.full(1, 0.0), x_bin_centers)
    log_p1, _ = interpolant.compute_log_pdf_and_score(np.full(1, 1.0), x_bin_centers)
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
    template = InterpolantPlotTemplate(
        figure=figure,
        data_pdf_axes=axes[0],
        interpolant_axes=axes[1],
        noise_pdf_axes=axes[2],
        schedule_axes=axes[3],
    )
    plot_schedule(template.schedule_axes, interpolant.noise_schedule)

    template.data_pdf_axes.plot(np.exp(log_p0), x_bin_centers, c="black")
    template.noise_pdf_axes.plot(np.exp(log_p1), x_bin_centers, c="black")
    template.data_pdf_axes.set_xlim(0, None)
    template.noise_pdf_axes.set_xlim(0, None)
    template.data_pdf_axes.set_xlabel("$p(x_0)$")
    template.noise_pdf_axes.set_xlabel("$p(x_1)$")
    template.data_pdf_axes.set_xticks([])
    template.interpolant_axes.yaxis.set_tick_params(labelleft=False)
    template.noise_pdf_axes.yaxis.set_tick_params(labelleft=False)
    template.schedule_axes.xaxis.set_tick_params(labelleft=False)
    template.noise_pdf_axes.set_xticks([])
    template.data_pdf_axes.set_ylabel("$x$")
    template.data_pdf_axes.grid()
    template.noise_pdf_axes.grid()
    template.schedule_axes.grid()
    template.interpolant_axes.set_xlim(*t_range)
    template.interpolant_axes.set_ylim(*x_range)
    template.interpolant_axes.set_aspect("auto")
    template.interpolant_axes.set_xlabel("$t$")
    template.figure.align_xlabels()
    return template


def plot_diffusion(output_path: str | None) -> InterpolantPlotTemplate:
    interpolant = GaussianMixtureInterpolant(
        noise_schedule=CosineNoiseSchedule(),
        x0_distr=GaussianMixture(
            means=np.array([-1.0, 2.0]),
            stddevs=np.array([0.1, 0.1]),
            weights=np.array([0.8, 0.2]),
        ),
    )
    x_range = (-3, 3)
    template = make_interpolant_plot_template(interpolant, x_range=x_range)
    plot_schedule(template.schedule_axes, interpolant.noise_schedule)
    plot_interpolant_pdf(template.interpolant_axes, interpolant, x_range=x_range)
    rng = np.random.default_rng(0)
    trajectory_t, trajectory_x = interpolant.sample_diffusion_trajectory(
        rng, nt=8192, nx=16
    )
    template.interpolant_axes.plot(
        trajectory_t, trajectory_x, c="white", linewidth=0.5, alpha=0.5
    )
    if output_path is not None:
        template.figure.savefig(os.path.join(output_path, "diffusion.png"))
    return template


def plot_probability_flow(
    noise_schedule: NoiseSchedule, output_path: str | None
) -> InterpolantPlotTemplate:
    interpolant = GaussianMixtureInterpolant(
        noise_schedule=noise_schedule,
        x0_distr=GaussianMixture(
            means=np.array([-1.0, 2.0]),
            stddevs=np.array([0.1, 0.1]),
            weights=np.array([0.8, 0.2]),
        ),
    )
    x_range = (-3, 3)
    template = make_interpolant_plot_template(interpolant, x_range=x_range)
    plot_schedule(template.schedule_axes, interpolant.noise_schedule)
    plot_interpolant_pdf(template.interpolant_axes, interpolant, x_range=x_range)
    trajectory_t, trajectory_x = interpolant.compute_probability_flow_trajectory(
        x_1=scipy.stats.norm.ppf(compute_bin_centers(0, 1.0, 16)),
        nt=2048,
    )
    template.interpolant_axes.plot(
        trajectory_t, trajectory_x, c="white", linewidth=1, alpha=0.5
    )
    if output_path is not None:
        template.figure.savefig(os.path.join(output_path, "probability_flow.png"))
    return template


def plot_slope_distribution(
    t_point: float, output_tag: str | None, output_path: str | None
) -> InterpolantPlotTemplate:
    interpolant = GaussianMixtureInterpolant(
        noise_schedule=LinearNoiseSchedule(),
        x0_distr=GaussianMixture(
            means=np.array([-1.0, 2.0]),
            stddevs=np.array([0.1, 0.1]),
            weights=np.array([0.8, 0.2]),
        ),
    )
    nt = 2048
    trajectory_t, trajectory_x = interpolant.compute_probability_flow_trajectory(
        x_1=scipy.stats.norm.ppf(compute_bin_centers(0, 1.0, 16)),
        nt=nt,
    )
    i_point, j_point = int(nt * (1 - t_point)), 12
    t_point = trajectory_t[i_point]
    x_t_point = trajectory_x[i_point, j_point]

    alpha_t_point = interpolant.noise_schedule.alpha(t_point)
    sigma_t_point = interpolant.noise_schedule.sigma(t_point)
    rng = np.random.default_rng(0)
    x0_line = interpolant.sample_x0_given_xt(
        rng=rng,
        size=50,
        t=np.full(1, t_point),
        x_t=np.full(1, x_t_point),
    )
    x1_line = (x_t_point - alpha_t_point * x0_line) / sigma_t_point
    lines = np.reshape(
        np.stack(
            [np.zeros_like(x0_line), x0_line, np.ones_like(x1_line), x1_line], axis=-1
        ),
        (-1, 2, 2),
    )
    template = make_interpolant_plot_template(interpolant, x_range=(-3, 3))
    plot_schedule(template.schedule_axes, interpolant.noise_schedule)
    template.interpolant_axes.plot(
        trajectory_t, trajectory_x, c="lightgray", linewidth=1.0
    )
    template.interpolant_axes.add_artist(
        matplotlib.collections.LineCollection(
            lines, linewidth=1.0, color="orangered", alpha=0.5
        )
    )
    template.interpolant_axes.plot(
        [t_point],
        [x_t_point],
        marker="o",
        markersize=3,
        markerfacecolor="k",
        markeredgecolor="k",
    )
    template.interpolant_axes.annotate(
        "$(t, x_t)$",
        (t_point, x_t_point),
        (0, 10),
        textcoords="offset points",
        ha="center",
        va="bottom",
    )
    template.interpolant_axes.plot(
        trajectory_t, trajectory_x[:, j_point], c="black", linewidth=1.5
    )
    if output_path is not None:
        assert output_tag is not None
        template.figure.savefig(
            os.path.join(output_path, f"slope_distribution_{output_tag}.svg"),
            format="SVG",
        )
    return template


def plot_consistency_models_integration(output_path: str):
    interpolant = GaussianMixtureInterpolant(
        noise_schedule=LinearNoiseSchedule(),
        x0_distr=GaussianMixture(
            means=np.array([-1.0, 2.0]),
            stddevs=np.array([0.1, 0.1]),
            weights=np.array([0.8, 0.2]),
        ),
    )

    nt = 2048
    trajectory_t, trajectory_x = interpolant.compute_probability_flow_trajectory(
        x_1=scipy.stats.norm.ppf(compute_bin_centers(0, 1.0, 8)),
        nt=nt,
    )

    t1 = 0.5
    i1, j1 = int(nt * (1 - t1)), 7
    t2 = 0.7
    i2, j2 = int(nt * (1 - t2)), 6
    t1 = trajectory_t[i1]
    t2 = trajectory_t[i2]
    x_t1 = trajectory_x[i1, j1]
    x_t2 = trajectory_x[i2, j2]
    x_01 = trajectory_x[-1, j1]
    x_02 = trajectory_x[-1, j2]
    line_x0 = (t2 * x_t1 - t1 * x_t2) / (t2 - t1)
    line_x1 = ((1 - t1) * x_t2 - (1 - t2) * x_t1) / (t2 - t1)
    template = make_interpolant_plot_template(interpolant, x_range=(-2, 4))
    plot_schedule(template.schedule_axes, interpolant.noise_schedule)
    template.interpolant_axes.plot(
        trajectory_t, trajectory_x, c="lightgray", linewidth=1.0
    )
    template.interpolant_axes.plot(
        trajectory_t[i1:], trajectory_x[i1:, j1], c="black", linewidth=1.5
    )
    template.interpolant_axes.plot(
        trajectory_t[i2:],
        trajectory_x[i2:, j2],
        c="black",
        linewidth=1.5,
        linestyle="--",
    )
    template.interpolant_axes.plot(
        [0, 1],
        [line_x0, line_x1],
        c="orangered",
        linewidth=1.0,
        marker="o",
        markersize=5,
        markerfacecolor="orangered",
        markeredgecolor="orangered",
    )
    template.interpolant_axes.add_artist(
        matplotlib.patches.FancyArrowPatch(
            posA=(trajectory_t[i1], trajectory_x[i1, j1]),
            posB=(trajectory_t[-1], trajectory_x[-1, j1]),
            arrowstyle="->",
            connectionstyle="arc3,rad=0.4",
            mutation_scale=15,
            shrinkA=10,
            shrinkB=10,
            edgecolor="royalblue",
            zorder=10,
        )
    )
    template.interpolant_axes.add_artist(
        matplotlib.patches.FancyArrowPatch(
            posA=(trajectory_t[i2], trajectory_x[i2, j2]),
            posB=(trajectory_t[-1], trajectory_x[-1, j2]),
            arrowstyle="->",
            connectionstyle="arc3,rad=-0.3",
            mutation_scale=15,
            shrinkA=10,
            shrinkB=10,
            edgecolor="royalblue",
            zorder=10,
        )
    )
    template.interpolant_axes.plot(
        [0, 0, t1, t2],
        [x_01, x_02, x_t1, x_t2],
        linestyle="none",
        marker="o",
        markersize=5,
        markerfacecolor="k",
        markeredgecolor="k",
    )
    template.interpolant_axes.annotate(
        "$x_0$",
        (0, line_x0),
        (5, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
    )
    template.interpolant_axes.annotate(
        "$x_1$",
        (1, line_x1),
        (-5, -5),
        textcoords="offset points",
        ha="right",
        va="top",
    )
    template.interpolant_axes.annotate(
        "$x_t$",
        (t1, x_t1),
        (0, 5),
        textcoords="offset points",
        ha="center",
        va="bottom",
    )
    template.interpolant_axes.annotate(
        r"$x_{t + \Delta t}$",
        (t2, x_t2),
        (0, 5),
        textcoords="offset points",
        ha="left",
        va="bottom",
    )
    template.interpolant_axes.annotate(r"$F_t$", (0.25, 2.7))
    template.interpolant_axes.annotate(r"$f_\theta$", (0.27, -0.1))
    if output_path is not None:
        template.figure.savefig(
            os.path.join(output_path, f"consistency_models_integration.svg"),
            format="SVG",
        )
    return template


def plot_p_and_score(output_path: str | None) -> matplotlib.figure.Figure:
    n_x0 = 384
    n_x1 = 256
    x0_mesh = compute_bin_centers(-3, 3, n_x0)
    x1_mesh = compute_bin_centers(-5, 5, n_x1)
    x0_mesh = np.broadcast_to(x0_mesh[:, None], (n_x0, n_x1))
    x1_mesh = np.broadcast_to(x1_mesh[None, :], (n_x0, n_x1))
    x_mesh = np.stack([x1_mesh, x0_mesh], axis=2)
    mean = np.array([[-3, -1], [1, 2]])
    var = np.square([2, 2.5])
    weights = np.array([0.4, 0.6])
    log_p, score = compute_multivariate_normal_log_p_and_score(
        x_mesh[:, :, None, :],
        mean=mean[None, None],
        var=var[None, None],
    )
    log_p = log_p + np.log(weights)
    log_p, score = compute_mixture_log_p_and_score(log_p, score)
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
    if output_path is not None:
        figure.savefig(os.path.join(output_path, "score_function.png"))
    return figure


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
    x_nodes = [lib.Node((spacing_x * i, 0), text, 17) for i, text in enumerate(x_text)]
    z_nodes = [
        lib.Node((spacing_x * (i + 1), 80), text, 17) for i, text in enumerate(z_text)
    ]
    zbar_nodes = [
        lib.Node((spacing_x * (i + 1), 80), text, 17)
        for i, text in enumerate(zbar_text)
    ]

    figure, axes = lib.make_graph_figure(np.array([[-20, -20], [310, 100]]))
    for node in x_nodes[:-1] + z_nodes:
        lib.plot_node(axes, node)
    for node1, node2 in zip(x_nodes[:-1], x_nodes[1:]):
        lib.plot_edge(axes, node1, node2)
    for node1, node2 in zip(z_nodes, x_nodes[1:]):
        lib.plot_edge(axes, node1, node2)
    figure.savefig(os.path.join(output_path, "forward_process.png"))

    figure, axes = lib.make_graph_figure(np.array([[-20, -20], [310, 100]]))
    for node in x_nodes[:-1] + zbar_nodes:
        lib.plot_node(axes, node)
    for node2, node1 in zip(x_nodes[:-1], x_nodes[1:]):
        lib.plot_edge(axes, node1, node2)
    for node1, node2 in zip(zbar_nodes, x_nodes[:-1]):
        lib.plot_edge(axes, node1, node2)
    figure.savefig(os.path.join(output_path, "backward_process.png"))


if __name__ == "__main__":
    task = sys.argv[1]
    output_path = os.path.join(lib.pages_dir(), "diffusion")
    match task:
        case "diffusion":
            plot_diffusion(output_path)
        case "probability_flow":
            plot_probability_flow(CosineNoiseSchedule(), output_path)
        case "p_and_score":
            plot_p_and_score(output_path)
        case "graphical_models":
            plot_graphical_models(output_path)
        case "slope_distribution":
            for i, t_point in enumerate([0.1, 0.4, 0.6, 0.9]):
                plot_slope_distribution(
                    t_point=t_point, output_tag=str(i), output_path=output_path
                )
        case "consistency_models_integration":
            plot_consistency_models_integration(
                os.path.join(lib.pages_dir(), "consistency_models")
            )
        case "consistency_models_probability_flow":
            plot_probability_flow(
                LinearNoiseSchedule(),
                os.path.join(lib.pages_dir(), "consistency_models"),
            )
        case _:
            raise ValueError(f"Unknown task: {task}")
