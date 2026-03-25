import numpy as np
from scipy.linalg import eig
from .hamiltonian import build_hamiltonian, diagonalize_hamiltonian
from .lattice_tools import initialize_positions, get_dimerized_chain
from .postprocessing import calculate_electronic_free_energy, calculate_static_lattice_energy

def center_of_mass_constraint(positions):
    return np.sum(positions)  # Queremos que sea igual a 0

def compute_energy(positions, eigenvalues, parameters):
    # Tu fórmula para la energía
    n_sites = parameters["number_of_sites"]
    W = parameters["cut_off_energy"]
    mu = parameters["changing_parameter"].real
    ee = calculate_electronic_free_energy(eigenvalues, mu , W)
    pe = calculate_static_lattice_energy(positions, parameters)
    tot_e = (ee + pe) / n_sites
    return   tot_e

def energy_to_minimize(positions, parameters):
    positions = (positions - np.mean(positions))
    H = build_hamiltonian(0, positions, parameters)
    eigenvalues, _, _ = diagonalize_hamiltonian(H, parameters)
    energy = compute_energy(positions, eigenvalues, parameters)
    return energy
   

