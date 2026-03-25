#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.constants as spc

def get_vector_potential(time, parameters):
    t_grid, A_t = parameters["external_perturbation"]
    vector_potential = np.interp(time, t_grid, A_t)
    return vector_potential

def gen_onsite_gates(potential_eV, width, separation, parameters, plus_minus=True):
    eV_to_Eh = 1/spc.physical_constants["Hartree energy in eV"][0] # Eh/eV
    n_sites = parameters["number_of_sites"]
    middle = (width + separation + width)
    if middle <= 1:
        width = int(width * n_sites)
        separation = int(separation * n_sites)
        middle = (width + separation + width)
    side = (n_sites - middle)//2
    if side < 0 or (side + middle + side) not in [n_sites, n_sites-1]:
        raise ValueError("Invalid width and separation!")
    onsite_gates = np.zeros(n_sites)
    if plus_minus: l_sign, r_sign = +1, -1
    else: l_sign, r_sign = -1, +1
    onsite_gates[(side):(side+width)] = l_sign * potential_eV * eV_to_Eh
    onsite_gates[::-1][(side):(side+width)] = r_sign * potential_eV * eV_to_Eh
    return onsite_gates

def get_onsite_energies(time, parameters):
    n_params = len(parameters["onsite_perturbation"])
    if n_params == parameters["number_of_sites"]:
        onsite_energies = parameters["onsite_perturbation"]
    elif n_params == 3:
        potential_eV, width, separation = parameters["onsite_perturbation"]
        if "onsite_plus_minus" not in parameters: plus_minus = True
        else: plus_minus = parameters["onsite_plus_minus"]
        onsite_energies = gen_onsite_gates(potential_eV, width, separation, parameters, plus_minus)
        parameters["onsite_perturbation"] = onsite_energies
    elif n_params == 2:
        t_grid, onsites_t = parameters["onsite_perturbation"]
        onsite_energies = np.interp(time, t_grid, onsites_t)
    else: raise RuntimeError("Provided format for onsite_perturbation is not allowed!")
    #if "uniform_onsite_noise" in parameters:
     #   onsite_energies += parameters["uniform_onsite_noise"]
    return onsite_energies

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
