#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.sparse as sp_sparse

def sp_init(matrix):
    M = sp_sparse.csr_matrix(matrix)
    return M

def sp_diag(array):
    M = sp_sparse.diags(array)
    return M

def sp_roll(matrix, shift):
    M = sp_init(matrix)
    rolled_M = sp_sparse.hstack((M[:,shift:], M[:,:shift]), format='csr')
    return rolled_M

###############################################################
#       Leandro Manuel Arancibia & Andrés Ignacio Bertoni     #
# (leandro.arancibia9@gmail.com)   (andresibertoni@gmail.com) #
###############################################################
