import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tabulate
from matplotlib import rc

def compute_coverage_1(gamma, mu):
    k = 1
    while mu > scipy.stats.chi2.ppf(1 - gamma, 2 * k) / 2:
        k += 1
    return scipy.stats.poisson.cdf(k - 1, mu)

def compute_coverage_2(gamma, mu):
    k = 0
    while mu > scipy.stats.chi2.ppf(gamma, 2 * (k + 1)) / 2:
        k += 1
    return 1 - scipy.stats.poisson.cdf(k - 1, mu)

def compute_coverage_3(gamma, mu):
    k1 = 0
    while mu > scipy.stats.chi2.ppf(gamma, 2 * (k1 + 1)) / 2:
        k1 += 1
    k2 = 1
    while mu > scipy.stats.chi2.ppf(1 - gamma, 2 * k2) / 2:
        k2 += 1
    return scipy.stats.poisson.cdf(k2 - 1, mu) - scipy.stats.poisson.cdf(k1 - 1, mu)


def plot_poisson_coverage():
    mu_values = np.arange(0.01, 7, 0.01)
    coverage_1 = [compute_coverage_1(0.95, mu) for mu in mu_values]
    coverage_2 = [compute_coverage_2(0.95, mu) for mu in mu_values]
    coverage_3 = [compute_coverage_3(0.95, mu) for mu in mu_values]


    figure, axes = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
    axes.plot(mu_values, coverage_1, label='$Q(J^{-}_{\\gamma}, \\lambda)$', linestyle='-')
    axes.plot(mu_values, coverage_2, label='$Q(J^{+}_{\\gamma}, \\lambda)$', linestyle='--')
    axes.plot(mu_values, coverage_3, label='$Q(J_{2\\gamma - 1}, \\lambda)$', linestyle='-.')
    axes.set_xlabel('$\\lambda$')
    axes.set_title('Coverage of Poisson intervals with $\\gamma = 0.95$')
    axes.axhline(0.9, linewidth=1, color='k', linestyle='dashed')
    axes.axhline(0.95, linewidth=1, color='k', linestyle='dashed')
    axes.legend()
    axes.grid()
    figure.tight_layout()
    figure.savefig('poisson_coverage.png')
    plt.close(figure)


def print_poisson_intervals():
    table = []
    for k in range(10):
        table.append([k])
        for gamma in [0.95, 0.975, 0.995]:
            table[-1].extend([
                scipy.stats.chi2.ppf(1 - gamma, 2 * k) / 2 if k > 0 else 0,
                scipy.stats.chi2.ppf(gamma, 2 * (k + 1)) / 2
            ])
    print("Poisson intervals:")
    print(tabulate.tabulate(table, tablefmt='simple', floatfmt='.2f'))

def print_students_t_critical_values():
    table = []
    for n in [2, 3, 4, 5, 10, 10_000, 100_000]:
        table.append([n])
        for gamma in [0.90, 0.95, 0.99]:
            table[-1].append(scipy.stats.t.ppf((1 + gamma) / 2, n - 1))
    print("Student t critical values:")
    print(tabulate.tabulate(table, tablefmt='simple', floatfmt='.2f'))

def plot_student_t_vs_n():
    t = np.linspace(-5, 5, 256)

    figure, axes = plt.subplots(1, 1, figsize=(5, 3), dpi=300)

    n_values = [2, 5, 10]
    linestyles = ['-', '--', '-.']
    for n, linestyle in zip(n_values, linestyles):
        pdf = scipy.stats.t.pdf(t, n - 1)
        axes.plot(t, pdf, label=f'$n={n}$', color='k', linestyle=linestyle, linewidth=1)
    axes.set_xlabel('$t$')
    axes.set_ylabel('PDF')
    axes.set_ylim(0, None)
    axes.set_xlim(-4.25, 4.25)
    axes.legend(framealpha=0)
    figure.tight_layout()
    figure.savefig('student_t_vs_n.png')
    plt.close(figure)


def plot_student_t_area():
    dof = 9
    t1 = np.linspace(-5, 5, 256)
    pdf1 = scipy.stats.t.pdf(t1, dof)

    t_90 = scipy.stats.t.ppf(0.95, dof)
    t2 = np.linspace(-t_90, t_90, 256)
    pdf2 = scipy.stats.t.pdf(t2, dof)

    t_95 = scipy.stats.t.ppf(0.975, dof)
    t3 = np.linspace(-t_95, t_95, 256)
    pdf3 = scipy.stats.t.pdf(t3, dof)

    figure, axes = plt.subplots(1, 1, figsize=(5, 3), dpi=300)
    axes.plot(t1, pdf1, color='black', linewidth=1)
    axes.fill_between(
        t2, pdf2, hatch='///', facecolor='none',
        edgecolor='k', linewidth=0.5, label='$\\gamma=0.9$')
    axes.fill_between(
        t3, pdf3, hatch='\\\\\\', facecolor='none',
        edgecolor='k', linewidth=0.5, label='$\\gamma=0.95$')
    arrowprops = {
        'facecolor': 'black',
        'arrowstyle': matplotlib.patches.ArrowStyle.Fancy(
            head_width=0.25,
            head_length=0.3,
            tail_width=0.02,
        )
    }
    axes.annotate(
        '$t_c(0.9)$', (t_90, 0),
        textcoords='offset points',
        xytext=(-45, -34.5),
        arrowprops=arrowprops,
    )
    axes.annotate(
        '$t_c(0.95)$', (t_95, 0),
        textcoords='offset points',
        xytext=(5, -34.5),
        arrowprops=arrowprops,
    )
    axes.set_xlabel('$t$')
    axes.set_ylabel('PDF')
    axes.set_ylim(0, None)
    axes.set_xlim(-4.25, 4.25)
    axes.legend(framealpha=0)
    figure.tight_layout()
    figure.savefig('student_t_area.png')
    plt.close(figure)


plt.style.use('../../_plot/mplstyle.rc')
matplotlib.rcParams['hatch.linewidth'] = 0.5

plot_poisson_coverage()
plot_student_t_area()
plot_student_t_vs_n()
print_poisson_intervals()
print_students_t_critical_values()
