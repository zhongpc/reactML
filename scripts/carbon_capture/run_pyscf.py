import argparse
import numpy as np

try:
    from gpu4pyscf import dft
except:
    from pyscf import dft
from pyscf import gto
from pyscf.hessian import thermo

from ase import Atoms, units
import ase.io
from sella import Sella

from reactML.common.utils import build_dft, dump_normal_mode
from reactML.common.ase_interface import PySCFCalculator


def hessian_function(atoms: Atoms, method: dft.rks.RKS | dft.uks.UKS) -> np.ndarray:
    """Calculate the Hessian matrix for the given atoms using the provided method."""
    method.mol.set_geom_(atoms.get_positions(), unit="Angstrom")
    method.run()
    hessian = method.Hessian().kernel()
    natom = method.mol.natm
    hessian = hessian.transpose(0, 2, 1, 3).reshape(3 * natom, 3 * natom)
    hessian *= (units.Hartree / units.Bohr**2)  # Convert from Hartree/Bohr^2
    return hessian


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xyzfile", type=str,
        help="Input xyz file",
    )
    parser.add_argument(
        "--xc", "-f", type=str, default="B3LYP",
        help="Name of Exchange-Correlation Functional",
    )
    parser.add_argument(
        "--basis", "-b", type=str, default="def2-SVPD",
        help="Name of Basis Set",
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
        "--disp", "-d", type=str, default=None,
        help="Type of Dispersion Correction",
    )
    parser.add_argument(
        "--grid", type=int, default=3,
        help="Grid level for numerical integration (0-9, default 3)",
    )
    parser.add_argument(
        "--scf-conv", type=float, default=1e-6,
        help="SCF convergence threshold (default 1e-6 a.u.)",
    )
    parser.add_argument(
        "--scf-max-cycle", type=int, default=200,
        help="Maximum number of SCF cycles (default 200)",
    )
    parser.add_argument(
        "--density-fit", "-ri", action="store_true",
        help="Whether to use density fitting (RI) approximation",
    )
    parser.add_argument(
        "--aux-basis", type=str, default=None,
        help="Auxiliary basis set for density fitting (default None, use default from pyscf)",
    )
    parser.add_argument(
        "--solvation", type=str, default=None,
        help="Type of solvation model (default None)",
    )
    parser.add_argument(
        "--solvent", type=str, default=None,
        help="Name of solvent for solvation model (default None)",
    )
    parser.add_argument(
        "--solvent-params", type=float, nargs="+", default=None,
        help="Parameters for solvation model (default None)",
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
        "--opt-fmax", type=float, default=4.5e-4,
        help="Maximum force for optimization convergence (default 4.5e-4 a.u.)",
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
        "--max-memory", type=int, default=None,
        help="Maximum memory in GB",
    )
    args = parser.parse_args()

    # read the xyz file
    mol = gto.Mole(charge=args.charge, spin=args.spin)  # mol.build() will be called in mol.fromfile
    mol.fromfile(filename=args.xyzfile)                 # we must set charge and spin before reading the file
    atoms = ase.io.read(args.xyzfile, format="xyz")

    mf = build_dft(mol, **vars(args))
    
    filename = args.xyzfile.rsplit(".", 1)[0]
    # geometric optimization
    if args.opt:
        calculator = PySCFCalculator(method=mf)
        atoms.calc = calculator
        trajectory = f"{filename}_opt.traj" if args.save_traj else None
        # sella_logfile = f"{filename}_sella.log" if args.save_traj else None
        opt = Sella(
            atoms=atoms,
            trajectory=trajectory,
            # logfile=sella_logfile,
            order=0,  # 0 for minimum, 1 for saddle point
            internal=args.internal,
            eig=args.calc_hessian,
            threepoint=True,
            diag_every_n=args.diag_every_n,
            hessian_function=lambda x: hessian_function(x, mf),
        )
        fmax = args.opt_fmax * units.Hartree / units.Bohr  # Convert from Hartree/Bohr
        opt.run(fmax=fmax, steps=args.opt_max_steps)
        # for step, _ in enumerate(opt.irun(fmax=fmax, steps=args.opt_max_steps)):
        #     hessian = opt.pes.H
        #     eigen_values = hessian.evals
        #     print(f"Step {step}: Hessian eigenvalues: {eigen_values}")
        # save the optimized geometry
        ase.io.write(f"{filename}_opt.xyz", opt.atoms, format="xyz")
        mf.mol.set_geom_(opt.atoms.get_positions(), unit="Angstrom")

    # single point calculation
    mf.run()

    # hessian calculation
    if args.freq:
        if not mf.converged:
            print("Warning: SCF calculation did not converge. Don't calculate frequencies.")
            return
        hessian = mf.Hessian().kernel()
        freq_info = thermo.harmonic_analysis(mf.mol, hessian, imaginary_freq=False)
        # imaginary frequencies
        freq_au = freq_info["freq_au"]
        if np.any(freq_au < 0):
            num_imag_freq = np.sum(freq_au < 0)
            print(f"Warning: {num_imag_freq} imaginary frequencies detected!")
        thermo_info = thermo.thermo(mf, freq_info["freq_au"], args.temp, args.press)
        dump_normal_mode(mf.mol, freq_info)
        thermo.dump_thermo(mf.mol, thermo_info)

        # exclude translation contributions
        G_tot_exclude_trans = thermo_info["G_tot"][0] - thermo_info["G_trans"][0]
        print("Gibbs free energy without translation contributions [Eh]", f"{G_tot_exclude_trans:.5f}")


if __name__ == "__main__":
    main()

