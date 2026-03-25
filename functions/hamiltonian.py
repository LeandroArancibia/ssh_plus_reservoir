#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .lattice_tools import get_sum_relative_positions
from .sparse_tools import sp_diag, sp_roll
from .perturbations import get_vector_potential, get_onsite_energies
from scipy.linalg import eig


def build_hamiltonian(time, positions, parameters):
    # Initialize diagonal and non-diagonal elements
    n_sites = parameters["number_of_sites"]
    H_ons = np.zeros(n_sites, dtype=complex)
    H_hop = np.zeros(n_sites, dtype=complex)
    # Electron-lattice coupling
    a = parameters["lattice_parameter"]
    hopping_params = parameters["hopping_parameters"]
    dr_n = get_sum_relative_positions(positions, [+1], parameters)
    for idx,t_i in enumerate(hopping_params):
        sign = (-1)**(idx+1) # Ensures electron-site attraction
        H_hop += sign * t_i * ((dr_n - a)**idx)
    # Includes coupling with external vector potential
    # (related to derivative of external electric field)
    if "external_perturbation" in parameters:
        vector_potential = get_vector_potential(time, parameters)
        H_hop *= np.exp(+1j * a * vector_potential)
    # Includes on-site perturbation
    if "onsite_perturbation" in parameters:
        onsite_energies = get_onsite_energies(time, parameters)
        H_ons += get_onsite_energies(time, parameters)
    # Builds sparse matrix
    # (non-diagonal elements)
    H = sp_diag(H_hop)
    H = sp_roll(H, shift=-1) # (upper diagonal)
    H = H + np.conjugate(H.T)
    # (diagonal elements)
    H += sp_diag(H_ons)
    return H

#def diagonalize_hamiltonian(hamiltonian, parameters):
# Only hermitian matrices
    #eigenvalues, eigenvectors = np.linalg.eig(hamiltonian.toarray())
    # (sorts states by energy)
    #energy_sort = np.argsort(eigenvalues)
    #state_energies = eigenvalues[energy_sort]
    #state_vectors = eigenvectors[:, energy_sort]
    # (ensures state vectors normalization)
    #state_vectors /= np.linalg.norm(state_vectors, axis=0)
    #state_vectors = state_vectors /  np.sqrt( np.sum( state_vectors.conj() * state_vectors, axis=0) )
    #return state_energies, state_vectors

def diagonalize_hamiltonian(hamiltonian, parameters):
    eigenvalues, eigenvectors_L, eigenvectors_R = eig(hamiltonian.toarray(), left=True, right=True)
    energy_sort = np.argsort(eigenvalues)
    eigvals = eigenvalues[energy_sort]
    left_vecs = eigenvectors_L[:,energy_sort]
    right_vecs = eigenvectors_R[:,energy_sort]
    weight = 1 /  np.sqrt( np.sum( left_vecs.conj() * right_vecs, axis=0))
    left_vecs *= weight.conj()
    right_vecs *=  weight 
    return eigvals, left_vecs, right_vecs


###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
