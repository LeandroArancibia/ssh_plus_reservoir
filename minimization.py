#!/usr/bin/env python
# coding: utf-8
import numpy as np
from scipy.optimize import minimize
import pickle
from functions.functions import energy_to_minimize, compute_energy
from functools import partial
from functions.lattice_tools import get_dimerized_chain
from functions.hamiltonian import build_hamiltonian, diagonalize_hamiltonian

parameters = {}
parameters["number_of_sites"] = 200
parameters["number_electrons_up"] = 100
parameters["number_electrons_down"] = 100
parameters["lattice_parameter"] = 1.22 
parameters["hopping_parameters"] = [2.5] # eV 
parameters["hopping_parameters"].append(4.1) # eV/A 
parameters["oscillator_parameters"] = [21.35] # eV/\AA^2
parameters["cut_off_energy"] = 15 # eV
parameters["periodic_boundaries"] = True
# gammas
gamma = 0
mu = 0
parameters["changing_parameter"] = (mu - 1.0j*gamma )
onsites_Gamma = np.zeros(parameters["number_of_sites"], dtype='complex')
onsites_Gamma[50:150] = (mu - 1.0j*gamma )
parameters["onsite_perturbation"] = onsites_Gamma

energies_per_iteration = []
def callback_function(current_positions):
    H = build_hamiltonian(0, current_positions, parameters)
    eigenvalues, _, _ = diagonalize_hamiltonian(H, parameters)
    energy = compute_energy(current_positions, eigenvalues, parameters)
    energies_per_iteration.append(energy)

# Creamos una versión "parcial" que fija el parámetro extra
energy_fn = partial(energy_to_minimize, parameters=parameters)

# Inicialización
initial_positions = get_dimerized_chain(0,200)
result = minimize(
    energy_fn,                   # Función a minimizar
    initial_positions,           # Guess inicial
    method='BFGS',              # Este método permite restricciones
    callback=callback_function        # Si querés guardar energía por iteración
)

resultados = {
    'changing_parameter': [],
    'energies_per_iteration': [],
    'optimization_results': []
}

resultados['changing_parameter'].append(parameters["changing_parameter"])
resultados['energies_per_iteration'].append(energies_per_iteration)
resultados['optimization_results'].append(result)

with open(f'results/BFGS_minimize_gamma_{gamma}_mu_{mu}.pkl', 'wb') as f:
    pickle.dump(resultados, f)

