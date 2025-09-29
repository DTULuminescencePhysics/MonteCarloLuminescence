from dataclasses import dataclass


@dataclass
class MetricPrefixes:
    G : float = 1e9    # Giga
    M : float = 1e6    # Mega
    k : float = 1e3    # Kilo
    c : float = 1e-2   # Centi
    m : float = 1e-3   # Milli
    u : float = 1e-6   # Micro
    n : float = 1e-9   # Nano
    ang : float = 1e-10 # Angstrom
    p : float = 1e-12  # Pico
    f : float = 1e-15  # Femto

    G2 : float = 1e18   # Giga^2
    M2 : float = 1e12   # Mega^2
    k2 : float = 1e6    # Kilo^2
    c2 : float = 1e-4   # Centi^2
    m2 : float = 1e-6   # Milli^2
    u2 : float = 1e-12  # Micro^2
    n2 : float = 1e-18  # Nano^2
    ang2 : float = 1e-20 # Angstrom^2
    p2 : float = 1e-24  # Pico^2
    f2 : float = 1e-30  # Femto^2

    G3 : float = 1e27   # Giga^3
    M3 : float = 1e18   # Mega^3
    k3 : float = 1e9    # Kilo^3
    c3 : float = 1e-6   # Centi^3
    m3 : float = 1e-9   # Milli^3
    u3 : float = 1e-18  # Micro^3
    n3 : float = 1e-27  # Nano^3
    ang3 : float = 1e-30 # Angstrom^3 
    p3 : float = 1e-36  # Pico^3
    f3 : float = 1e-45  # Femto^3

mp = MetricPrefixes()


@dataclass(frozen=True)
class physical_constants:
    k_b: float = 1.380649e-23  # Boltzmann constant in J/K (kg·m²/s²·K)
    k_b_ev: float = 8.617333262e-5  # Boltzmann constant in eV/K
    h: float = 6.62607015e-34   # Planck constant in J·s (kg·m²/s)
    h_bar: float = 1.054571817e-34 # Reduced Planck constant in J·s (kg·m²/s)
    c: float = 299792458        # Speed of light in m/s
    e: float = 1.602176634e-19  # Elementary charge in C
    m_e: float = 9.10938356e-31 # Electron mass in kg
    N_A: float = 6.02214076e23  # Avogadro's number in 1/mol
    R: float = 8.314462618  # Gas constant in J/(mol·K)
    epsilon_0: float = 8.854187817e-12  # Vacuum permittivity in F/m 
    ev_to_j: float = 1.602176634e-19  # Conversion factor from eV to J
    
cnst = physical_constants()


