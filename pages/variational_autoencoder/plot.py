import matplotlib.pyplot as plt
import numpy as np

def tangent_parabula(a: float, x0: float, x: np.ndarray):
    b = 2 * (1 - a) * x0
    c = (1 - a) * x0 * x0 - b * x0
    return a * np.square(x) + b * x + c


def plot_example_losses():
    figure, axes = plt.subplots(
       1, 1, figsize=(5.0, 3.25), dpi=300,
       gridspec_kw={
           'left': 0.07,
           'right': 0.99,
           'top': 0.99,
           'bottom': 0.10,
       },
    )
    x = np.linspace(-1, 1, 128)
    x0 = 0.5
    l_true = np.square(x)
    l_approx_1 = tangent_parabula(2.0, x0, x)
    l_approx_2 = tangent_parabula(1.8, x0, x) + 0.15
    axes.axvline(0, color='gray')
    axes.axvline(x0, color='gray')
    axes.plot(
        x, l_approx_2,
        label=r'$L(q(\cdot|x, \phi), p(\cdot|x, \theta))$',
        color='black',
        linestyle='-.',
    )
    axes.plot(
        x, l_approx_1,
        label=r'$L(p(\cdot|x, \theta_0), p(\cdot|x, \theta))$',
        color='black',
        linestyle='--',
    )
    axes.plot(
        x, l_true, 
        label=r'$-\log p(x|\theta)$',
        color='black',
    )
    axes.spines.top.set_visible(False)
    axes.spines.right.set_visible(False)
    axes.spines[:].set(linewidth=1.5)
    axes.xaxis.set_tick_params(width=1.5)
    axes.set_ylabel('Loss')
    axes.set_xlim(-0.5, 1)
    axes.set_ylim(-0.1, 0.7)
    axes.set_yticks([])
    axes.set_xticks([0, x0, 0.9])
    axes.set_xticklabels([
        r'$\theta_{\mathrm{ML}}$',
        r'$\theta_0$',
        r'$\theta$',
        ])
    axes.legend(
        loc=(0.01, 0.61),
        fancybox=False,
        framealpha=1,
    )
    figure.savefig('example_losses.png')
    plt.close(figure)

plt.style.use('../../_plot/mplstyle.rc')
plot_example_losses()

