# Self-Consistent Minimization of a Surface-Coupled Polyacetylene Chain

This repository contains the code used to obtain the results presented in the following preprint:

https://arxiv.org/abs/2603.15835

## Overview

This code implements a self-consistent energy minimization scheme for a polyacetylene chain coupled to a surface. The approach is based on a tight-binding description combined with a path integral formalism, as detailed in the reference article.

The main script, `minimization.py`, provides a fully self-contained example for a chain of 200 sites interacting with a surface.

## Methodology

The minimization algorithm is based on the BFGS optimization routine and proceeds as follows:

1. An initial configuration for the positions of the CH groups is defined.
2. The system Hamiltonian is constructed, including the coupling to the surface.
3. The resulting (generally non-Hermitian) Hamiltonian is diagonalized.
4. Complex eigenvalues are obtained and used to compute the total energy.
5. The atomic positions are updated according to the BFGS minimization scheme.
6. Steps 2–5 are iterated until convergence is achieved according to a predefined tolerance.

The energy functional being minimized is derived from the path integral formalism described in the reference work.

## Notes

The Hamiltonian is non-Hermitian due to the coupling with the surface, leading to complex eigenvalues.

## Outputs

Once convergence is reached, the code provides:

- Equilibrium positions of the CH groups
- Minimum energy of the system

## Features and Flexibility

The implementation allows for flexible modification of several physical parameters, including:

- Number of sites in the chain
- Coupling points to the surface
- Strength of the surface coupling
- Boundary conditions (open or periodic)
- Hopping parameters

## Physical Regime

All calculations are performed at zero temperature.

## Usage

Run the main script:

```bash
python minimization.py
```

## Requirements

- Python 3.12.3
- NumPy
- SciPy

## Reference

If you use this code, please cite:
https://arxiv.org/abs/2603.15835

