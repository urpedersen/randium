from math import log

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


betas = [0.20, 0.40, 0.60, 0.80, 0.90, 1.00, 1.10, 1.20, 1.24, 1.30, 1.32]

t_halfs = []

plt.figure()
plt.title(r'$D=2$, $L=64$, $M=512$, $N_m=8$, MC-local. Fit: $Q(t)=\exp(-(t/\tau)^\gamma)$')
plt.plot([0.1, 1e8], [0.5, 0.5], 'k:')
p0 = [0.8, 0.8]
data = []
for beta in betas:
    filename = f'data/overlap_{beta:.2f}.csv'
    df = pd.read_csv(filename)
    t = df['times']
    Q = df['overlap']
    # Fit to exp
    y_fit_min, y_fit_max = 0.2, 0.8
    def func(t, tau, gamma):
        return np.exp(-(t/tau) ** gamma)
    selection = Q>=y_fit_min
    selection &= Q<=y_fit_max
    popt, pcov = curve_fit(func, t[selection], Q[selection], p0=p0)
    p0 = popt
    tau = popt[0]
    gamma = popt[1]
    t_half = tau*np.log(2)**(1/gamma)
    t_halfs.append(t_half)
    t_fit = np.logspace(-1, 5, 128)
    plt.plot(t, Q, 'o', label=r'$\beta=$' f'{beta:.2f}, ' r'$\tau=$' f'{tau:.1e}, ' r'$\gamma=$' f'{gamma:.2f}, ' r'$t_½=$' f'{t_half:.2e}')
    plt.plot(t_fit, func(t_fit, *popt), 'k--')
    data.append({'beta': beta, 't_half': t_half})

    print(popt)
plt.xlim(0.1, 1e8)
plt.ylim(0, 1)
plt.xlabel(r'Time, $t$ [MC-global attempts per particle]')
plt.ylabel(r'Overlap, $Q(t)$')
plt.legend(fontsize='6')
plt.xscale('log')
plt.savefig('figures/overlap_vs_time.png', dpi=300)
plt.show()

# Load MC-global t_halfs
df_global = pd.read_csv('data/overlap_global.csv')
Q0 = log(2)/2

# Plot t_halfs
plt.figure()
plt.title(r'$D=2$, $L=64$, $M=512$, $N_m=8$')
plt.plot(betas, t_halfs, 'bo', label='MC-local')
plt.plot(df_global['beta'], df_global['t_half'], 'rd', label='MC-global')
plt.plot([0], [Q0], 'ko', label='ln(2)/2')
plt.yscale('log')
plt.ylabel(r'Half-overlap time, $t_½$')
plt.xlabel(r'$\beta=1/T$')
plt.ylim(0.1, 1e4)
plt.xlim(0, 1.4)
plt.grid()
plt.legend()
plt.savefig('figures/overlap_t_halfs.png', dpi=300)
plt.show()

pd.DataFrame(data).to_csv(
    'data/overlap_local.csv',
    index=False)
