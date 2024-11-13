import time
import pyscf
import numpy as np
import os
from pyscf import lib
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


import numpy as np
from ase import Atoms
import os

def perturb_atoms(atoms, amplitude=0.01):
    """
    Add random perturbations to atomic positions in an ASE Atoms object.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        The atomic structure to perturb
    amplitude : float, optional
        Maximum displacement amplitude in Angstroms (default: 0.01)
        
    Returns:
    --------
    ase.Atoms
        New Atoms object with perturbed positions
    """
    
    # Create a copy of the original atoms object
    perturbed_atoms = atoms.copy()
    
    # Get the number of atoms
    n_atoms = len(atoms)
    
    # Generate random displacements for each atom in 3D
    # Using normal distribution with mean 0 and std = amplitude/3
    # (this ensures most perturbations are within ±amplitude)
    displacements = np.random.normal(0, amplitude/3, size=(n_atoms, 3))
    
    # Get original positions
    positions = perturbed_atoms.get_positions()
    
    # Add displacements to positions
    new_positions = positions + displacements
    
    # Set new positions
    perturbed_atoms.set_positions(new_positions)
    
    return perturbed_atoms





def get_mfGPU(mol):

    xc = 'wb97xd'
    # auxbasis = 'def2-svp-jkfit' #  need to find the right auxbasis
    # auxbasis = 'def2-tzvpp-jkfit'
    scf_tol = 1e-10
    max_scf_cycles = 200
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

def hessian_function(atoms):
    atom_string = ase_to_string(atoms)
    mol = pyscf.M(atom=atom_string, basis= basis, max_memory= max_memory)
    mf_GPU = get_mfGPU(mol)

    # Compute the Hessian
    h = mf_GPU.Hessian() 
    h.auxbasis_response = 2

    h_dft = h.kernel()
    natm = h_dft.shape[0]
    h_dft_reshape = h_dft.copy()
    h_dft_reshape = h_dft_reshape.transpose([0,2,1,3]).reshape([3*natm,3*natm])

    return h_dft_reshape
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--order', type=int, default= 0, help='Order of the saddle point, 0 for optimized geometry, 1 and above for saddle point')
    parser.add_argument('--dir', type=str, default='./data/structures_3point_develope/INPUT_TS8.xyz', help='DFT functional')
    parser.add_argument('--fmax', type=float, default=0.001, help='Maximum force for optimization')
    parser.add_argument('--steps', type=int, default=500, help='Number of optimization steps')
    parser.add_argument('--delta0', type=float, default=0.1, help='Initial trust radius')
    parser.add_argument('--basis', type=str, default='def2-svpd', help='Basis set')
    parser.add_argument('--max_memory', type=int, default=32000, help='Maximum memory')
    args = parser.parse_args()

    basis = args.basis
    max_memory = args.max_memory
    delta0 = args.delta0
    threepoint = True

    atom_path = os.path.join(args.dir, 's.xyz')
    atoms = read(atom_path)

    atoms = perturb_atoms(atoms, amplitude = 0.05)
    
    atoms_string = ase_to_string(atoms)
    calculator = PySCF_calculator(mf_class = get_mfGPU)

    atoms.set_calculator(calculator)

    print(atoms.get_potential_energy())
    print(atoms.get_forces())

    execute_path = os.path.join(args.dir, 'execution_opt.log')
    logfile_path = os.path.join(args.dir, 'sella_opt.log')
    traj_path = os.path.join(args.dir, 'sella_opt.traj')

    if os.path.exists(logfile_path):
        os.remove(logfile_path)
    if os.path.exists(traj_path):
        os.remove(traj_path)
    if os.path.exists(execute_path):
        os.remove(execute_path)


    # Configure logging
    logging.basicConfig(
        filename = execute_path,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )



    # Create Sella optimizer
    opt = Sella(
        atoms,
        gamma = 0.1,             # convergence criterion for iterative diagonalization
        delta0 = delta0,         # initial trust radius
        logfile= logfile_path,
        order = args.order,
        threepoint = threepoint,
        # sigma_inc = 1.15, # default to be 1.15
        # sigma_dec = 0.65, # default to be 0.65
        diag_every_n = 40,
        nsteps_per_force_diag = 30, # when force is small, update the Hessian every n steps
        check_nsteps = [5, 10, 20], # update the Hessian at the given steps
        eta = 1e-6,
        hessian_function = hessian_function
    )

    # Create trajectory file
    traj = Trajectory(traj_path , 'w')

    # Get initial energy and forces
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    fmax = max(np.abs(forces.flatten()))

    print("Initial config:")
    print(f"Energy: {energy:.6f}")
    print(f"Max force: {fmax:.6f}")

    start_time = time.time()

    # Run optimization iteratively
    before_next_hessian = 0

    for step, _ in enumerate(opt.irun(fmax= args.fmax, steps= args.steps)):
        # Save current configuration to trajectory
        atoms_tosave = atoms.copy()
        atoms_tosave.calc = None
        traj.write(atoms_tosave)

        if opt.delta < 1e-4:
            print("Optimization converged with minmum delta (trust radius)!")
            opt.delta = 5e-4

        hessian = opt.pes.H
        eigen_values = hessian.evals
        print(eigen_values)

    # Save final configuration
    traj.close()

    print("\nOptimization completed!")
    elapsed_time = time.time() - start_time
    logging.info(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    atoms.write(os.path.join(args.dir, 'final.xyz'))

    logging.info(f"Eigenvalues of the Hessian matrix:\n")
    logging.info(np.sort(eigen_values))
