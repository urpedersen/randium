from matplotlib import pyplot as plt

import randium as rd

L = 64
Ms = [8, 16, 128, 512]
betas = 0.6, 0.8, 1.0, 1.2, 1.4, 1.5, 2.0, 3.0, 8.0
lats = []
# Set-up lattices
for i in range(len(Ms)):
    lat = rd.Lattice(L, Ms[i], 2, betas[0])
    lats.append(lat)
    print(f'N={lat.N}, M={lat.M}, N_m={lat.N_m}')

# Equilibrate lattices
eq_steps = 2048*lats[0].N
for lat in lats:
    for beta in betas:
        lat.beta = beta
        lat.simulation_monte_carlo_global(eq_steps)
    print()

plt.figure(figsize=(8, 8))
for idx, lat in enumerate(lats):
    arr = lat.get_types_on_lattice()
    arr.resize((lat.L, lat.L))
    plt.subplot(2,2,idx+1)
    plt.axis('off')
    plt.title(
        f'M={lat.M}, N_m={lat.N_m}'
    )
    plt.imshow(arr, cmap='hsv')
plt.savefig('figures/ground_state.png',dpi=300)
plt.show()
