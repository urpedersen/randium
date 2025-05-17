
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
cm = mpl.colormaps['jet_r']


plt.figure()

for beta_old in 1.0, 1.1, 1.2, 1.4, 1.5, 1.6:
    color_idx = (beta_old-0.8)/0.8
    color = cm(color_idx)
    print(f'{beta_old = }, {color_idx = }')
    #beta_old = 1.50
    beta_new = 1.30
    fname=f'data/ageing_{beta_old:.4f}_to_{beta_new:.4f}.npz'
    data = np.load(fname)
    times = data['times']
    energy_data = data['energy_data']

    u = energy_data.mean(axis=0)
    du = energy_data.std(axis=0)
    t = times

    def shifted_stretch_exponential(x, alpha, tau, gamma):
        B = -2.0*beta_new
        A = -2.0*beta_old - B
        return alpha*A*np.exp(-(x/tau) ** gamma)+B
    p0 = 0.98, 1e4, 0.5
    t_fit = np.logspace(0, 8, 128)
    popt, pcov = curve_fit(shifted_stretch_exponential, t, u, p0=p0)
    print(f'{p0 = }, {popt = }')
    gamma = popt[2]
    tau = popt[1]

    plt.errorbar(t, u, du, fmt='o', label=r'$\beta_i=$' f'{beta_old} (' r'$\tau=$' f'{tau:.1}' r'; $\gamma=$' f'{gamma:.2})', color=color)
    #plt.plot(t, u, 'o', label=f'{beta_old} -> {beta_new}')
    plt.plot(t_fit, shifted_stretch_exponential(t_fit, *popt), '--', color=color)

plt.text(1e6, -2.55, r'$\beta_f=1.3$', color='k', fontsize=14)
plt.text(4e4, -3.1, r'Fits: $A\,\exp(-(t/\tau)^\gamma)-2\beta_f$', color='k', fontsize=12)
plt.xlabel(r'Time, $t$ [swap attempts per particle]')
plt.ylabel(r'Energy, $u=E/N$')
plt.xscale('log')
plt.xlim(1, 1e8)
plt.ylim(-3.2, -1.8)
plt.legend(frameon=False, fontsize='10')
plt.savefig('figures/ageing.png', dpi=300)
plt.show()
