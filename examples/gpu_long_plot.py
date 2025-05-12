from pprint import pprint

import toml

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import curve_fit

# Import jet color mat
cm = mpl.colormaps['jet_r']


L=192
betas=[0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]


  #####################
  ##  Plot overlaps  ##
  #####################

def stretch_exponential(x, A, tau, gamma):
    return A*np.exp(-(x/tau) ** gamma)
p0 = 1.0, 10, 0.9
t_fit = np.logspace(0, 7, 128)

plt.figure()
plt.title(f'Randium on a {L}x{L} lattice ' r'($N_m=1$)')
t_halfs = []
for idx, beta in enumerate(betas):
    fname=f'local_{L}x{L}M36864beta{beta:.4f}'
    print(fname)
    lattice = np.load(f'data/{fname}.npy')
    data = toml.load(f'data/{fname}.toml')
    print(data.keys())
    t = data['times']
    Q = data['overlaps']
    beta = float(fname.split('beta')[-1])
    popt, pcov = curve_fit(stretch_exponential, t, Q, p0=p0)
    t_half = popt[1]*np.log(2*popt[0])**(1/popt[2])
    t_halfs.append(t_half)
    print(t_half, popt)
    #plt.plot([t_half], [0.5], 'kx', color=cm(idx/6))
    p0 = popt
    plt.plot(t_fit, stretch_exponential(t_fit, *popt), 'k--')
    plt.plot(t, Q, 'o', label=r'$\beta=$' f'{beta:.2f}', color=cm(idx/8))
plt.text(1e6, 0.43, r'Fit: $A\,\exp(-(t/\tau)^\gamma)$', color='k')
wall_clock_one_minute = 2e8/192**2*60
plt.annotate('1 minute on\nthe wall-clock',
             xy=(wall_clock_one_minute, 0.7),
             xytext=(wall_clock_one_minute, 0.9),
             fontsize=10,
             arrowprops=dict(arrowstyle='->', lw=1),
             ha='center',
             va='center',
             color='k'
             )
plt.xlabel(r'Time, $t$')
plt.ylabel(r'Overlap, $Q$')
plt.xscale('log')
plt.ylim(0, 1.0)
plt.xlim(1, 1e8)
plt.legend(frameon=False, fontsize=10, labelspacing=0.1, loc='upper right')
plt.savefig('figures/overlap_vs_time.png', dpi=300)
plt.show()


  ###########################
  #  Plot relaxation times  #
  ###########################

def fit_quadratic(x, tau, beta_0, J):
    return tau*np.exp(J**2*(x-beta_0)**2)

beta_fit = np.linspace(0.0, 0.8, 10)
t_theory = np.log(2)/2.0*np.exp(2.0*beta_fit)
plt.figure()
plt.plot(betas, t_halfs, 'o', color='k')
plt.plot(beta_fit, t_theory, 'r--')
plt.text(0.18, 2, r'$\frac{\log(2)}{2}\exp(2\beta)$', fontdict={'size': 16}, color='r')
# Fit Parabolic scaling
beta_fit = np.linspace(1.1, 1.7, 32)
popt, pcov = curve_fit(fit_quadratic, betas, t_halfs, p0=[1.0, 0.6, 4.0])
print('Parabolic scaling: ',popt)
plt.plot(beta_fit, fit_quadratic(beta_fit, *popt), 'b--')
plt.text(1.0, 2, r'$\tau_0\,\exp(J^2(\beta-\beta_0)^2)$', color='b', fontdict={'size': 16})
plt.ylim(1e-1, 1e6)
plt.xlim(0.0, 1.7)
plt.yscale('log')
plt.xlabel(r'Inverse temperature, $\beta$')
plt.ylabel(r'Half-time, $t_\frac{1}{2}$')
plt.xlim(0, None)
plt.savefig('figures/t_halfs.png', dpi=300)
plt.show()

  ###########################################
  # Plot overlap vs. scaled relaxation time #
  ###########################################

plt.figure()
plt.title(f'Randium on a {L}x{L} lattice ' r'($N_m=1$)')
for idx, beta in enumerate(betas):
    fname=f'local_{L}x{L}M36864beta{beta:.4f}'
    lattice = np.load(f'data/{fname}.npy')
    data = toml.load(f'data/{fname}.toml')
    t = data['times']
    Q = data['overlaps']
    beta = float(fname.split('beta')[-1])
    plt.plot(t/t_halfs[idx], Q, 'o', label=r'$\beta=$' f'{beta:.2f}', color=cm(idx/8))
plt.text(1e6, 0.43, r'Fit: $A\,\exp(-(t/\tau)^\gamma)$', color='k')
x_fit = np.logspace(-4, 3, 128)
A, gamma = 0.98, 0.5
tau = 1.0/(np.log(2*A))**(1/gamma)
plt.plot(x_fit, stretch_exponential(x_fit, A, tau, gamma), 'r--', label=r'$0.98\,\exp(-\sqrt{\tilde t})$')
plt.xlabel(r'Scaled Time, $\tilde t = t/t_\frac{1}{2}$')
plt.ylabel(r'Overlap, $Q$')
plt.xscale('log')
plt.ylim(0, 1.0)
plt.xlim(1e-4, 1e3)
plt.legend(frameon=False, fontsize=10, labelspacing=0.1, loc='upper right')
plt.savefig('figures/overlap_vs_time_scaled.png', dpi=300)
plt.show()

  ##################
  ## Mean energy  ##
  ##################

enrs = []
for idx, beta in enumerate(betas):
    fname=f'local_{L}x{L}M36864beta{beta:.4f}'
    lattice = np.load(f'data/{fname}.npy')
    data = toml.load(f'data/{fname}.toml')
    us = data['energies']
    enrs.append(np.mean(us[-10:]))
    print(len(us))
plt.figure(figsize=(6, 4))
plt.plot(betas, enrs, 'ko')
beta_fit = np.linspace(0.0, 1.7, 10)
plt.plot(beta_fit, -2*beta_fit, 'b--')
plt.text(1.15, -2.0, r'$\langle u\rangle = -2\beta$', fontdict={'size': 20}, color='b')
plt.xlabel(r'Inverse temperature, $\beta$')
plt.ylabel(r'Mean energy, $\langle u \rangle$')
plt.xlim(0, 1.7)
plt.ylim(-3.5, 0.0)
plt.savefig('figures/mean_energy.png', dpi=300)
plt.show()


