import argparse
from types import SimpleNamespace

import numpy as np
from mace.calculators import mace_omol
from ase import units
import ase.io
from sella import Sella
from pyscf import gto, symm
from pyscf.hessian import thermo

from reactML.common.utils import dump_normal_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xyzfile", type=str,
        help="Input xyz file",
    )
    parser.add_argument(
        "--model", type=str,
        help="Path to the MACE model file",
    )
    parser.add_argument(
        "--device", type=str, default="",
        help="Device to use for MACE (e.g., 'cuda' for GPU, default is GPU if available, otherwise CPU)",
    )
    parser.add_argument(
        "--charge", "-c", type=int, default=0,
        help="Total charge"
    )
    parser.add_argument(
        "--spin", "-s", type=int, default=0,
        help="Total spin (2S not 2S+1)",
    )
    parser.add_argument(
        "--opt", "-o", action="store_true",
        help="Whether to do optimize the geometry",
    )
    parser.add_argument(
        "--internal", action="store_true",
        help="Whether to use internal coordinates for optimization",
    )
    parser.add_argument(
        "--calc-hessian", action="store_true",
        help="Whether to calculate the Hessian matrix during optimization",
    )
    parser.add_argument(
        "--diag-every-n", type=int, default=None,
        help="Number of steps per diagonalization in optimization (default 3)",
    )
    parser.add_argument(
        "--opt-fmax", type=float, default=0.05,
        help="Maximum force for optimization convergence (default 0.05 eV/Å)",
    )
    parser.add_argument(
        "--opt-max-steps", type=int, default=1000,
        help="Maximum number of optimization steps (default 1000)",
    )
    parser.add_argument(
        "--save-traj", action="store_true",
        help="Whether to save the optimization trajectory",
    )
    parser.add_argument(
        "--freq", action="store_true",
        help="Whether to calculate the frequencies",
    )
    parser.add_argument(
        "--temp", "-T", type=float, default=298.15,
        help="Temperature for thermodynamic analysis",
    )
    parser.add_argument(
        "--press", "-p", type=float, default=101325.,
        help="Pressure for thermodynamic analysis",
    )
    parser.add_argument(
        "--symm-geom-tol", type=float, default=1e-5,
        help="Symmetry geometry tolerance (default 1e-5)",
    )
    args = parser.parse_args()

    # set symmetry tolerance
    symm.geom.TOLERANCE = args.symm_geom_tol
    
    atoms = ase.io.read(args.xyzfile, format="xyz")
    atoms.info["charge"] = args.charge
    atoms.info["spin"] = args.spin + 1  # Convert PySCF's 2S (args.spin) to ASE's 2S+1 by adding 1
    atoms.calc = mace_omol(model=args.model)
    n_atoms = len(atoms)

    filename = args.xyzfile.rsplit(".", 1)[0]

    # Run MACE optimization
    if args.opt:
        trajectory = f"{filename}_opt.traj" if args.save_traj else None
        opt = Sella(
            atoms,
            trajectory=trajectory,
            order=0,  # 0 for minimum, 1 for saddle point
            internal=args.internal,
            eig=args.calc_hessian,
            threepoint=True,
            diag_every_n=args.diag_every_n,
            hessian_function=lambda x: x.calc.get_hessian().reshape(n_atoms * 3, n_atoms * 3)
        )
        opt.run(fmax=args.opt_fmax, steps=args.opt_max_steps)
        ase.io.write(f"{filename}_opt.xyz", atoms, format="xyz")

    energy = atoms.get_potential_energy()
    print(f"MACE energy: {energy:.6f} eV  ({energy / units.Hartree:.6f} Eh)")

    if args.freq:
        hessian = atoms.calc.get_hessian()
        hessian = hessian.reshape(n_atoms, 3, n_atoms, 3).transpose(0, 2, 1, 3)
        hessian *= (units.Bohr**2 / units.Hartree)  # Convert from eV/Ang^2 to Hartree/Bohr^2
        # create a temporary Mole()
        mol = gto.M(
            atom=[(ele, coord) for ele, coord in zip(atoms.get_chemical_symbols(), atoms.get_positions())],
            charge=args.charge,
            spin=args.spin,
        )
        freq_info = thermo.harmonic_analysis(mol, hessian, imaginary_freq=False)
        # imaginary frequencies
        freq_au = freq_info["freq_au"]
        if np.any(freq_au < 0):
            num_imag_freq = np.sum(freq_au < 0)
            print(f"Warning: {num_imag_freq} imaginary frequencies detected!")
        dummy_mf = SimpleNamespace(mol=mol, e_tot=energy / units.Hartree)
        thermo_info = thermo.thermo(dummy_mf, freq_au, args.temp, args.press)
        dump_normal_mode(mol, freq_info)
        thermo.dump_thermo(mol, thermo_info)

        # exclude translation contributions
        G_tot_exclude_trans = thermo_info["G_tot"][0] - thermo_info["G_trans"][0]
        print("Gibbs free energy without translation contributions [Eh]", f"{G_tot_exclude_trans:.5f}")


if __name__ == "__main__":
    main()