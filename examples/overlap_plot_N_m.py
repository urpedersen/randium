import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

N_ms = [16, 8, 4, 2, 1]
beta = 1.24

plt.figure()
plt.title(r'$D=2$, $L=64$, $\beta=1.24$, MC-local. Fit: $Q(t)=\exp(-(t/\tau)^\gamma)$')
plt.plot([0.1, 1e8], [0.5, 0.5], 'k:')
p0 = [0.95, 1000, 0.7]
data = []
for N_m in N_ms:
    filename = f'data/overlap_{beta:.2f}_{N_m}.csv'
    df = pd.read_csv(filename)
    t = df['times']
    Q = df['overlap']
    # Fit to exp
    y_fit_min, y_fit_max = 0.2, 0.9
    def func(t, A, tau, gamma):
        return A*np.exp(-(t/tau) ** gamma)
    selection = Q>=y_fit_min
    selection &= Q<=y_fit_max
    popt, pcov = curve_fit(func, t[selection], Q[selection], p0=p0)
    p0 = popt
    A = popt[0]
    tau = popt[1]
    gamma = popt[2]
    t_half = tau*np.log(2)**(1/gamma)
    t_fit = np.logspace(-1, 5, 128)
    plt.plot(t, Q, 'o', label=r'$N_m=$' f'{N_m}, ' r'$A=$' f'{A:.4}, ' r'$\tau=$' f'{tau:.1e}, ' r'$\gamma=$' f'{gamma:.2f}, ' r'$t_½=$' f'{t_half:.2e}')
    plt.plot(t_fit, func(t_fit, *popt), 'k--')
    data.append({'beta': beta, 't_half': t_half})

    print(popt)
plt.xlim(0.5, 1e5)
plt.ylim(0, 1.4)
plt.xlabel(r'Time, $t$ [MC-local attempts per particle]')
plt.ylabel(r'Overlap, $Q(t)$')
plt.legend(fontsize='8')
plt.xscale('log')
plt.savefig('figures/overlap_vs_time_N_m.png', dpi=300)
plt.show()
