#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from .sparse_tools import sp_roll

def initialize_positions(parameters):
    n_sites = parameters["number_of_sites"]
    a = parameters["lattice_parameter"]
    positions = a * np.arange(n_sites)
    if "initialize_positions_noise" in parameters:
        noise = parameters["initialize_positions_noise"]
        if "random_seed" in parameters: np.random.seed(seed=parameters["random_seed"])
        positions += (noise/100) * np.random.uniform(-1, 1, n_sites) * a
    # (sum_n r_n = 0)
    positions += - np.mean(positions)
    return positions

def get_neighbour_matrix(neighbours, parameters):
    n_sites = parameters["number_of_sites"]
    a = parameters["lattice_parameter"]
    is_periodic = parameters["periodic_boundaries"]
    N = np.zeros((n_sites, n_sites))
    for nidx in neighbours:
        n = np.ones(n_sites)
        if not is_periodic: n[::np.sign(nidx)][-abs(nidx):] = 0
        N += sp_roll(np.diag(n), -nidx).toarray()
    return N

def get_paired_matrix(neighbours, parameters):
    N = get_neighbour_matrix(neighbours, parameters)
    paired_matrix = N - np.diag(np.sum(N, axis=-1))
    return paired_matrix

def get_sum_relative_positions(positions, neighbours, parameters):
    n_sites = parameters["number_of_sites"]
    a = parameters["lattice_parameter"]
    is_periodic = parameters["periodic_boundaries"]
    P = get_paired_matrix(neighbours, parameters)
    sum_rel_positions = P @ positions
    if is_periodic:
        for neig_idx in neighbours:
            correction = np.sign(neig_idx) * a * n_sites
            sum_rel_positions[::np.sign(neig_idx)][-abs(neig_idx):] += correction
    return sum_rel_positions

def positions_to_bondlengths(positions, parameters):
    is_periodic = parameters["periodic_boundaries"]
    neighbours = [+1]
    bondlengths = np.abs(get_sum_relative_positions(positions, neighbours, parameters))
    if not is_periodic: bondlengths = bondlengths[:-1]
    return bondlengths

def check_lattice_minimum(old_positions, new_positions, parameters):
    tolerance = parameters["lattice_optimization_tolerance"]
    is_periodic = parameters["periodic_boundaries"]
    old_bondlengths = positions_to_bondlengths(old_positions, parameters)
    new_bondlengths = positions_to_bondlengths(new_positions, parameters)
    if is_periodic: new_bondlengths = new_bondlengths[::-1]
    bondlength_diffs = new_bondlengths - old_bondlengths
    rmse = np.sqrt(np.mean(bondlength_diffs**2))
    is_optimized = (rmse <= tolerance)
    # If necessary, checks for two alternating minima   
    if not is_optimized:
        parameters["rmse"] = rmse # to know how much the error is
        if "__old_old_positions" in parameters.keys():
            old_old_positions = parameters["__old_old_positions"]
            old_old_bondlengths = positions_to_bondlengths(old_old_positions, parameters)
            bondlength_diffs = new_bondlengths - old_old_bondlengths
            rmse = np.sqrt(np.mean(bondlength_diffs**2))
            is_optimized = (rmse <= tolerance)
        if is_optimized:
            print("Found second minimum!\n(Stored in parameters as 'alt_opt_positions'!)")
            parameters["alt_opt_positions"] = new_positions
    parameters["__old_old_positions"] = old_positions
    if is_optimized: del parameters["__old_old_positions"]
    return is_optimized
    
def get_dimerized_chain(since ,till ):
    #Dimerized chain with 200 sites
    print('Dimerized chain in Agnstrom')
    Rn = np.array([-121.55999063, -120.16936584, -119.05813603 ,-117.73865911 ,-116.60634159,
 -115.30018406, -114.16132591, -112.86008974, -111.71847457, -110.41946094,
 -109.27652283, -107.97860459, -106.83499972, -105.537653 ,  -104.39368299,
 -103.09664848, -101.95247739, -100.65561751,  -99.51134058 , -98.21457595,
  -97.07024082,  -95.77352381,  -94.62915164,  -93.33246638 , -92.18807833,
  -90.89140895,  -89.74701032,  -88.45035152,  -87.30594759,  -86.00929409,
  -84.86487958,  -83.56823666,  -82.42382215,  -81.12717923,  -79.98275942,
  -78.68612179,  -77.54170199,  -76.24505907,  -75.10064456,  -73.80400164,
  -72.65958713,  -71.36293892,  -70.21851911,  -68.92188148,  -67.77746168,
  -66.48082405,  -65.33639896,  -64.03976133,  -62.89534153,  -61.5987039,
  -60.45428145,  -59.15764118,  -58.01322402,  -56.71658374,  -55.57216394,
  -54.27552102,  -53.13110651,  -51.83446359,  -50.69004379,  -49.39340298,
  -48.24898635,  -46.95234343,  -45.80792363,  -44.51128389,  -43.3668662,
  -42.07022593,  -40.92580348,  -39.62916585,  -38.48474604,  -37.18810312,
  -36.04368332,  -34.74704569,  -33.60262589,  -32.30598562,  -31.16156634,
  -29.86492554,  -28.72050574,  -27.42386652,  -26.27944672,  -24.98280697,
  -23.83838717,  -22.54174795,  -21.39732815,  -20.10068787,  -18.95626807,
  -17.6596278 ,  -16.51520799,  -15.21856719,  -14.07414898,  -12.77750764,
  -11.63308943,  -10.33644809,   -9.19202988,   -7.89538855,   -6.75096954,
   -5.454329  ,   -4.30990972,   -3.01326913,   -1.86885018,   -0.57220948,
    0.57220948,    1.86885018,    3.01326913,    4.30990972,    5.454329,
    6.75096954,    7.89538855,    9.19202988,   10.33644809,   11.63308943,
   12.77750764,   14.07414898,   15.21856719,   16.51520799,   17.6596278,
   18.95626807,   20.10068787,   21.39732815,   22.54174795,   23.83838717,
   24.98280697,   26.27944672,   27.42386652,   28.72050574,   29.86492554,
   31.16156634 ,  32.30598562,   33.60262589,   34.74704569,   36.04368332,
   37.18810312,   38.48474604,   39.62916585,   40.92580348,   42.07022593,
   43.3668662 ,   44.51128389,   45.80792363,   46.95234343,   48.24898635,
   49.39340298,   50.69004379,   51.83446359,   53.13110651,   54.27552102,
   55.57216394,   56.71658374,   58.01322402,   59.15764118,   60.45428145,
   61.5987039 ,   62.89534153,   64.03976133,   65.33639896,   66.48082405,
   67.77746168,   68.92188148,   70.21851911,   71.36293892,   72.65958713,
   73.80400164,   75.10064456,   76.24505907,   77.54170199,   78.68612179,
   79.98275942,   81.12717923,   82.42382215,   83.56823666,   84.86487958,
   86.00929409,   87.30594759,   88.45035152,   89.74701032,   90.89140895,
   92.18807833,   93.33246638 ,  94.62915164,   95.77352381,   97.07024082,
   98.21457595,   99.51134058,  100.65561751,  101.95247739,  103.09664848,
  104.39368299,  105.537653  ,  106.83499972,  107.97860459,  109.27652283,
  110.41946094,  111.71847457,  112.86008974,  114.16132591,  115.30018406,
  116.60634159,  117.73865911,  119.05813603,  120.16936584,  121.55999063,  ])
    return Rn[since:till]
###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
