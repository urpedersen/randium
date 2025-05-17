import math
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

data = np.loadtxt('2d_local/t_half.csv', delimiter=',', skiprows=1)
betas = data[:, 0]
t_halfs = data[:, 1]
t_0 = math.log(2)/2
log_t_halfs = np.log(t_halfs/t_0)
mask = betas > 1.2


def fit_func(x, J):
    a = 10
    x_0 = 0.7
    S = 1/(1+np.exp(-a*(x-x_0)))  # sigmoid switch, logistic function
    return (1-S)*2*x + S*(J**2*(x-x_0)**2+2*x)
popt, pcov = curve_fit(fit_func, betas, log_t_halfs)
print(popt)

J = 3.5

plt.figure()
plt.plot(betas, np.exp(log_t_halfs), 'o')
x_fit = np.linspace(0, 1.8, 64)
plt.plot(x_fit, np.exp(fit_func(x_fit, J)), '--')
plt.xlabel(r'$\beta$')
plt.ylabel(r'$t_\mathrm{half}$')
plt.savefig('t_half.png', dpi=300)
plt.yscale('log')
plt.show()

