import time
import pyscf
import numpy as np
import os
import json
from pyscf import lib
from pyscf.hessian import thermo
from gpu4pyscf.dft import rks
from monty.json import jsanitize
# from gpu4pyscf import dft
# from pyscf.geomopt.geometric_solver import optimize

from ase.io import read, write
from ase import Atoms
from ase.io import read, write
from ase.io import Trajectory

from sella import Sella
from reactML.common.pyscf2ase import ase_to_string, PySCF_calculator

import argparse
import logging

lib.num_threads(8)




def get_mfGPU(mol):
    xc = 'wb97xd'
    # auxbasis = 'def2-svp-jkfit' #  need to find the right auxbasis
    # auxbasis = 'def2-tzvpp-jkfit'
    scf_tol = 1e-10
    max_scf_cycles = 500
    screen_tol = 1e-14
    grids_level = 3
        
    mol.verbose = 1
    mf_GPU = rks.RKS(mol, xc=xc, disp='d3bj').density_fit()
    mf_GPU.grids.level = grids_level
    mf_GPU.conv_tol = scf_tol
    mf_GPU.max_cycle = max_scf_cycles
    mf_GPU.screen_tol = screen_tol

    mf_GPU = mf_GPU.PCM()
    mf_GPU.verbose = 1 # 6 for details
    mf_GPU.grids.atom_grid = (99,590)
    mf_GPU.small_rho_cutoff = 1e-10
    mf_GPU.with_solvent.lebedev_order = 29 # 302 Lebedev grids
    mf_GPU.with_solvent.method = 'COSMO' # 'IEF-PCM' # C-PCM, SS(V)PE, COSMO
    mf_GPU.with_solvent.eps = 78.3553 # for water
    # mf_GPU.with_solvent.method = 'IEF-PCM'
    # mf_GPU.with_solvent.solvent = 'water'
    mf_GPU.kernel()
    
    return mf_GPU


def get_mfGPU_metaGGA(mol):
    xc = 'HYB_MGGA_XC_WB97M_V'
    # auxbasis = 'def2-svp-jkfit' #  need to find the right auxbasis
    # auxbasis = 'def2-universal-jkfit'
    scf_tol = 1e-10
    max_scf_cycles = 100
    screen_tol = 1e-14
    grids_level = 3
        
    mol.verbose = 1
    mf_GPU = rks.RKS(mol, xc=xc, disp= None).density_fit()
    mf_GPU.grids.level = grids_level
    mf_GPU.conv_tol = scf_tol
    mf_GPU.max_cycle = max_scf_cycles
    mf_GPU.screen_tol = screen_tol

    mf_GPU = mf_GPU.PCM()
    mf_GPU.verbose = 1 # 6 for details
    mf_GPU.grids.atom_grid = (99,590)
    mf_GPU.nlcgrids.atom_grid = (50,194)
    mf_GPU.small_rho_cutoff = 1e-10
    mf_GPU.with_solvent.lebedev_order = 29 # 302 Lebedev grids
    mf_GPU.with_solvent.method = 'COSMO' # 'IEF-PCM' # C-PCM, SS(V)PE, COSMO
    mf_GPU.with_solvent.eps = 78.3553 # for water
    # mf_GPU.with_solvent.method = 'IEF-PCM'
    # mf_GPU.with_solvent.solvent = 'water'
    mf_GPU.kernel()
    
    return mf_GPU




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--order', type=int, default= 1, help='Order of the saddle point, 0 for optimized geometry, 1 and above for saddle point')
    parser.add_argument('--dir', type=str, default='./', help='DFT functional')
    parser.add_argument('--fmax', type=float, default=1e-4, help='Maximum force for optimization')
    parser.add_argument('--steps', type=int, default=500, help='Number of optimization steps')
    parser.add_argument('--basis', type=str, default='def2-svpd', help='Basis set')
    parser.add_argument('--max_memory', type=int, default=32000, help='Maximum memory')
    args = parser.parse_args()

    basis = args.basis
    max_memory = args.max_memory


    atom_path = os.path.join(args.dir, 'final.xyz')
    atoms = read(atom_path)
    atoms_string = ase_to_string(atoms)

    mol = pyscf.M(atom=atoms_string, basis= basis, max_memory= max_memory)

    mf_GPU = get_mfGPU(mol) # for Hessian: should not be the metaGGA functional

    # Compute Hessian
    h = mf_GPU.Hessian()
    h.auxbasis_response = 2
    h_dft = h.kernel()

    # harmonic analysis
    thermo_analysis = thermo.harmonic_analysis(mol, h_dft)
    thermo.dump_normal_mode(mol, thermo_analysis)

    thermo_analysis = thermo.thermo(mf_GPU, thermo_analysis['freq_au'], 298.15, 101325)
    thermo.dump_thermo(mol, thermo_analysis)

    with open(os.path.join(args.dir, 'thermo_analysis.json') , 'w') as f:
        json.dump(jsanitize(thermo_analysis), f)
