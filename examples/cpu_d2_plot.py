import toml
import matplotlib.pyplot as plt


Ls = 8, 16, 32


plt.figure(figsize=[8,4])
for L in Ls:
    M = L*L
    filename = f'data/local_{L}x{L}M{M}beta1.3000_cpu.toml'
    print(f'Loading {filename}')
    data = toml.load(filename)
    print(f'{data = }')
    plt.plot(data['times'], data['overlaps'], '-o', label=f'L={L}')
plt.xscale('log')
plt.legend()
plt.show()
