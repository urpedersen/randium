import toml

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.optimize import curve_fit

# Import jet color mat
cm = mpl.colormaps['jet_r']

L=96
M=L*L

  ##############
  #  Overlaps  #
  ##############

def stretch_exponential(x, A, tau, gamma):
    return A*np.exp(-(x/tau) ** gamma)
p0 = 1.0, 1, 0.9
t_fit = np.logspace(-4, 4, 128)

betas=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
t_halfs = []
for idx, beta in enumerate(betas):
    fname = f'local_{L}x{L}M{M}beta{beta:.4f}_cpu'
    data = toml.load(f'data/{fname}.toml')
    t = np.array(data['times'])
    Q = np.array(data['overlaps'])
    mask = (0.2 < Q) & (Q < 0.8)
    beta = data['beta']
    popt, pcov = curve_fit(stretch_exponential, t[mask], Q[mask], p0=p0)
    t_half = popt[1]*np.log(2*popt[0])**(1/popt[2])
    print(f'{beta},{t_half}')
    t_halfs.append(t_half)
    p0 = popt
    plt.plot(t_fit, stretch_exponential(t_fit, *popt), '--', color=cm(idx/8))
    plt.plot(t, Q, 'o', label=r'$\beta=$' f'{beta:.2f}', color=cm(idx/8))

plt.xscale('log')
plt.show()
