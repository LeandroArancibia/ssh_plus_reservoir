#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.constants as spc

def get_occupations(state_energies, parameters):
    # (occupations may be a given parameter)
    if "occupations" in parameters.keys():
        return parameters["occupations"]
    # Fermi-Dirac + Aufbau
    # Boltzmann constant in atomic units
    if "electronic_temperature" in parameters:
        k_B = spc.physical_constants["Boltzmann constant in eV/K"][0] # eV/K
        T = parameters["electronic_temperature"]
        kT = k_B * T # eV
        Eh_to_eV = spc.physical_constants["hartree-electron volt relationship"][0]
        kT /= Eh_to_eV # Eh
    else: kT = 0
    # Occupation according to Fermi-Dirac distribution
    occupations = np.zeros((state_energies.size, 2))
    for jdx,spin in enumerate(["up", "down"]):
        N_spin = parameters[f"number_electrons_{spin}"]
        # Compute Fermi energy
        homo_energy_s = state_energies[N_spin-1]
        lumo_energy_s = state_energies[N_spin]
        fermi_energy_s = np.mean([homo_energy_s, lumo_energy_s])
        if kT > 0:
            weights = np.exp((state_energies - fermi_energy_s)/kT)
            occupations[:,jdx] = 1/(weights + 1)
        else: occupations[:,jdx] = np.where(state_energies < fermi_energy_s, 1, 0)
        # Ensure conservation of number of electrons
        occupations[:,jdx] *= N_spin / np.sum(occupations[:,jdx])
    # Returns occupations for given electr. temperature
    return occupations

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
