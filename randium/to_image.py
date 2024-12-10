""" Make png images for a movie """
import randium as rd
import matplotlib.pyplot as plt




def main():
    lat = rd.default_lattice()
    fig, ax = get_fig(lat)
    fig.show()

if __name__ == '__main__':
    main()
