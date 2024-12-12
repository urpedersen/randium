import matplotlib.pyplot as plt

import randium as rd

def f2s(num):
    """ format_two_significant_digits """
    if num == 0:
        return "0"
    # Calculate the number of digits before the decimal point
    magnitude = int(f'{abs(num):e}'.split('e')[1])
    return f'{num:.{2 - magnitude}f}' if magnitude < 2 else f'{round(num, -(magnitude - 1))}'

betas = [0.0, 0.6, 1.1, 1.4]
lats = []  # Lattices of the initial configuration
#betas = [0.0]*4

plt.figure(figsize=(8, 8))
for idx, beta in enumerate(betas):
    plt.subplot(2,2,idx+1)
    plt.axis('off')
    lat = rd.Lattice(64, 512, 2, beta=beta)
    steps_eq = 2048*lat.N  # 2048*lat.N
    lat.simulation_monte_carlo_global(steps_eq)
    lat_ref = lat.copy()
    overlap = 1.0
    steps = 0
    steps_per_update = 512
    while overlap >= 0.5:
        lat.simulation_monte_carlo_local(steps_per_update)
        steps += steps_per_update
        overlap, overlaps = lat.overlap(lat_ref)
    t = steps/lat.N
    print(beta, t, overlap)
    plt.title(r'$\beta=$' f'{beta}, ' r'$t_½=$' f'{f2s(t)}')
    overlaps.resize((lat.L, lat.L))
    plt.imshow(overlaps, cmap='gray_r')
    lats.append(lat)
plt.savefig('figures/overlap_half.png',dpi=300)
plt.show()


# Show the
plt.figure(figsize=(8, 8))
for idx, (lat, beta) in enumerate(zip(lats, betas)):
    plt.subplot(2,2,idx+1)
    plt.axis('off')
    arr = lat.get_types_on_lattice()
    arr.resize((lat.L, lat.L))
    plt.imshow(arr, cmap='hsv')
    plt.title(r'$\beta=$' f'{beta}')
plt.savefig('figures/overlap_half_types.png',dpi=300)
plt.show()
