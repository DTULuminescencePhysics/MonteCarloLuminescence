import matplotlib.pyplot as plt
import numpy as np
from classes.physics import crystal
from classes.constants import mp

def probability_function(x,rho):
    return np.exp((-4*np.pi*rho*np.power(x,3))/3)*(4*np.pi*rho*np.power(x,2))

def unitless_r(r,rho):
    return np.cbrt(4*np.pi*rho/3)*r

def unitless_probability_function(r):
    return 3*np.square(r)*np.exp(-np.power(r,3))


def probability_plot(r,rho):
    rho1 = rho
    r_plot =r/mp.n
    for i in range(5):
        y = probability_function(r, rho1)
        l=i+20-6
        y=y*1e-10
        plt.plot(r_plot,y,label=f"$ρ=10^{{{l}}}μm^{{-3}}$")
        rho1 *=10
    plt.title('Nearest Neighbour Probability Distribution')
    plt.xlabel('Distance r (nm)')
    plt.ylabel('Probability Density p(r) (Å⁻¹)')
    plt.legend()
    plt.xscale('log')
    plt.xlim(1,1000)
    plt.ylim(0,0.020)
    plt.show()

def unitless_probability_plot(r,rho):
    rho1 = rho
    for i in range(5):
        r_unitless = unitless_r(r,rho1)
        y = unitless_probability_function(r_unitless)
        plt.plot(r_unitless,y)
        rho1 *=10

    plt.title('Nearest Neighbour Probability Distribution')
    plt.xlabel("Distance unitless r'")
    plt.ylabel("Probability Density p(r')")
    plt.xlim(0,2.0)
    plt.show()


def nearest_neighbour_histogram(distances1):
    distance = distances1
    rounded_distances = distance/mp.ang
    # rounded_distances = [round(d,10) for d in distance]
    plt.hist(rounded_distances, bins='auto', edgecolor='black')
    rho1 = 1e25 #7000/30*mp.n3
    # rho1 = 2.3e28
    r = np.linspace(0,10*mp.ang,1000000)
    r_plot=r/mp.ang
    y = probability_function(r, rho1)
    y=y*mp.u
    plt.plot(r_plot,y)#,label=f"$ρ=10^{{{l}}}μm^{{-3}}$")
    # distance2 = distance2*mp.ang
    # rounded_distances2 = [round(d,2) for d in distances2]

    # plt.hist(rounded_distances2, bins='auto', edgecolor='blue')

    plt.title('Histogram of Nearest Neighbour Distances')
    plt.xlabel('Distance (Å)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def random_sampling():
    length = 30*mp.ang
    width = 40*mp.ang
    height = 25*mp.ang
    n_0 =7000
    trap_0 = 7000
    test_crystal = crystal(Height=height, Width=width, Length=length, boundary_factor=0, n_0=n_0, trap_0=trap_0)
    test_crystal.initialise_electrons()
    test_crystal.initialise_traps()
    distances1 = test_crystal.nearest_neighbours()
    nearest_neighbour_histogram(distances1)




def main():
    start =0
    end = mp.u
    r = np.linspace(start,end,1000000)
    rho = 1e20
    random_sampling()
    # probability_plot(r,rho)
    # unitless_probability_plot(r,rho)


if __name__ == "__main__":
    main()
    
    
    



