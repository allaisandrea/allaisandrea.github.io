# %%
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import numpy
import tabulate
from matplotlib import rc

def compute_coverage_1(alpha, mu):
    k = 1
    while mu > scipy.stats.chi2.ppf(1 - alpha, 2 * k) / 2:
        k += 1
    return scipy.stats.poisson.cdf(k - 1, mu)

def compute_coverage_2(alpha, mu):
    k = 0
    while mu > scipy.stats.chi2.ppf(alpha, 2 * (k + 1)) / 2:
        k += 1
    return 1 - scipy.stats.poisson.cdf(k - 1, mu)

def compute_coverage_3(alpha, mu):
    k1 = 0
    while mu > scipy.stats.chi2.ppf(alpha, 2 * (k1 + 1)) / 2:
        k1 += 1
    k2 = 1
    while mu > scipy.stats.chi2.ppf(1 - alpha, 2 * k2) / 2:
        k2 += 1
    return scipy.stats.poisson.cdf(k2 - 1, mu) - scipy.stats.poisson.cdf(k1 - 1, mu)

plt.style.use('../../_plot/mplstyle.rc')

mu_values = numpy.arange(0.01, 7, 0.01)
coverage_1 = [compute_coverage_1(0.95, mu) for mu in mu_values]
coverage_2 = [compute_coverage_2(0.95, mu) for mu in mu_values]
coverage_3 = [compute_coverage_3(0.95, mu) for mu in mu_values]


figure, axes = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
axes.plot(mu_values, coverage_1, label='$Q(J^{-}_{\\alpha}, \\lambda)$', linestyle='-')
axes.plot(mu_values, coverage_2, label='$Q(J^{+}_{\\alpha}, \\lambda)$', linestyle='--')
axes.plot(mu_values, coverage_3, label='$Q(J_{2\\alpha - 1}, \\lambda)$', linestyle='-.')
axes.set_xlabel('$\\lambda$')
axes.set_title('Coverage of Poisson intervals with $\\alpha = 0.95$')
axes.axhline(0.9, linewidth=1, color='k', linestyle='dashed')
axes.axhline(0.95, linewidth=1, color='k', linestyle='dashed')
axes.legend()
axes.grid()
figure.savefig('poisson_coverage.png')
plt.close(figure)

table = []
for k in range(10):
    table.append([])
    for alpha in [0.95, 0.975, 0.995]:
        table[-1].extend([
            k, scipy.stats.chi2.ppf(1 - alpha, 2 * k) / 2 if k > 0 else 0,
            scipy.stats.chi2.ppf(alpha, 2 * (k + 1)) / 2
        ])
print(tabulate.tabulate(table, tablefmt='latex', floatfmt='.3f'))
