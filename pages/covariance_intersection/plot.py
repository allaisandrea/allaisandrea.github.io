import numpy
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib


def make_random_spd_matrix(rng, n, scale):
    entries = rng.normal(size=n * (n + 1) // 2)
    symm = numpy.zeros((n, n))
    k = 0
    for i in range(n):
        for j in range(i, n):
            symm[i, j] = entries[k]
            symm[j, i] = entries[k]
            k += 1
    return scipy.linalg.expm(scale * symm)


def is_spd(M):
    return numpy.linalg.eigvalsh(M)[0] > 0


def plot_covariance_ellipse(axes, covariance, **kwargs):
    s2, u = numpy.linalg.eigh(covariance)
    axes.add_patch(
        matplotlib.patches.Ellipse(
            (0, 0),
            2 * numpy.sqrt(s2[0]),
            2 * numpy.sqrt(s2[1]),
            angle=numpy.arctan2(u[0, 1], u[0, 0]) * 180 / numpy.pi,
            **kwargs,
        ),
    )


plt.style.use('../../_plot/mplstyle.rc')
rng = numpy.random.default_rng(17)
P = make_random_spd_matrix(rng, 4, 1)
Paa = P[:2, :2]
Pab = P[:2, 2:]
Pbb = P[2:, 2:]
K = numpy.concatenate([numpy.identity(2)] * 2)
Pdd = numpy.linalg.inv(numpy.transpose(K) @ numpy.linalg.inv(P) @ K)
n_Pcc = 6
Pcc_list = [
    numpy.linalg.inv(w * numpy.linalg.inv(Paa) + (1 - w) * numpy.linalg.inv(Pbb))
    for w in (numpy.arange(0, n_Pcc) + 0.5) / n_Pcc
]
figure, axes = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
for i, Pcc in enumerate(Pcc_list):
    plot_covariance_ellipse(
        axes,
        Pcc,
        facecolor="none",
        edgecolor="tab:blue",
        label="$P_{cc}$" if i == 0 else None,
        linestyle="dashed",
    )
plot_covariance_ellipse(
    axes, Paa, facecolor="none", edgecolor="k", label="$P_{aa}$, $P_{bb}$"
)
plot_covariance_ellipse(axes, Pbb, facecolor="none", edgecolor="k")
plot_covariance_ellipse(
    axes,
    Pdd,
    facecolor="none",
    edgecolor="tab:orange",
    linestyle="dashdot",
    label="$P_{dd}$",
)
axes.legend()
plot_range = 1.1 * numpy.sqrt(
    [numpy.amax([Paa[0, 0], Pbb[0, 0]]), numpy.amax([Paa[1, 1], Pbb[1, 1]])]
)
axes.set_xlim(-plot_range[0], plot_range[0])
axes.set_ylim(-plot_range[1], plot_range[1])
axes.set_aspect("equal")
axes.set_xticks([])
axes.set_yticks([])
axes.axvline(0, color="k", linewidth=0.75)
axes.axhline(0, color="k", linewidth=0.75)
figure.tight_layout()
plt.savefig("geometric_interpretation.png")
plt.close(figure)
