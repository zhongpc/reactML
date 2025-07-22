import argparse
import numpy as np
from pyscf import gto
from pyscf.geomopt import geometric_solver
from pyscf.hessian import thermo

from reactML.common.utils import write_xyz, build_dft


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
        "--basis", "-b", type=str, default="6-31++G(d,p)",
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
        help="SCF convergence threshold (default 1e-6)",
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
        "--opt-max-cycle", type=int, default=200,
        help="Maximum number of optimization cycles (default 200)",
    )
    parser.add_argument(
        "--opt-conv-energy", type=float, default=1e-6,
        help="Convergence threshold for optimization energy (default 1e-6)",
    )
    parser.add_argument(
        "--opt-conv-grms", type=float, default=3e-4,
        help="Convergence threshold for optimization gradient RMS (default 3e-4 Eh/Bohr)",
    )
    parser.add_argument(
        "--opt-conv-gmax", type=float, default=4.5e-4,
        help="Convergence threshold for optimization gradient MAX (default 4.5e-4 Eh/Bohr)",
    )
    parser.add_argument(
        "--opt-conv-drms", type=float, default=1.2e-3,
        help="Convergence threshold for optimization displacement RMS (default 1.2e-3 Angstrom)",
    )
    parser.add_argument(
        "--opt-conv-dmax", type=float, default=1.8e-3,
        help="Convergence threshold for optimization displacement MAX (default 1.8e-3 Angstrom)",
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

    # read the xyz file with pyscf
    mol = gto.Mole()
    mol.fromfile(filename=args.xyzfile)

    # build the molecule with C-PCM solvation model
    mf = build_dft(mol, **vars(args))

    # geometric optimization
    if args.opt:
        opt = geometric_solver.GeometryOptimizer(mf)
        opt.params = {
            'convergence_energy': args.opt_conv_energy,  # Eh
            'convergence_grms': args.opt_conv_grms,  # Eh/Bohr
            'convergence_gmax': args.opt_conv_gmax,  # Eh/Bohr
            'convergence_drms': args.opt_conv_drms,  # Angstrom
            'convergence_dmax': args.opt_conv_dmax,  # Angstrom
        }
        opt.max_cycle = args.opt_max_cycle
        opt.kernel()
        if not opt.converged:
            print("Geometry Optimization not converged!!")
            return
        # save optimized struct
        optfile = args.xyzfile.replace(".xyz", "_opt.xyz")
        write_xyz(mf.mol, optfile)

    # single point calculation
    mf.run()

    # hessian calculation
    if args.freq:
        hessian = mf.Hessian().kernel()
        freq_info = thermo.harmonic_analysis(mf.mol, hessian, imaginary_freq=False)
        # imaginary frequencies
        freq_au = freq_info["freq_au"]
        if np.any(freq_au < 0):
            num_imag_freq = np.sum(freq_au < 0)
            print(f"Warning: {num_imag_freq} imaginary frequencies detected!")
        thermo_info = thermo.thermo(mf, freq_info["freq_au"], args.temp, args.press)
        thermo.dump_normal_mode(mf.mol, freq_info)
        thermo.dump_thermo(mf.mol, thermo_info)

        # exclude translation contributions
        G_tot_exclude_trans = thermo_info["G_tot"][0] - thermo_info["G_trans"][0]
        print("Gibbs free energy without translation contributions [Eh]", f"{G_tot_exclude_trans:.5f}")


if __name__ == "__main__":
    main()

