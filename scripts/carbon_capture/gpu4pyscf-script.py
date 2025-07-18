import os.path as osp
import argparse
import numpy as np
from pyscf import gto
from gpu4pyscf import dft
from pyscf.geomopt import geometric_solver
from pyscf.hessian import thermo


def write_xyz(mol: gto.Mole, filename: str, charge: int = 0, spin: int = 1):
    elements = mol.elements
    coords = mol.atom_coords(unit="Angstrom")
    with open(filename, "w") as f:
        f.write(f"{len(elements)}\n")
        # Convert spin (2S) to multiplicity (2S+1) format, as required by quantum chemistry conventions.
        f.write(f"{charge} {spin + 1}\n")
        for ele, coord in zip(elements, coords):
            f.write(f"{ele: <2}    {coord[0]:10.6f}    {coord[1]:10.6f}    {coord[2]:10.6f}\n")


def read_pcm_eps():
    # from https://gaussian.com/scrf/
    pcm_eps_txt = osp.join(osp.dirname(__file__), "pcm_eps.txt")
    with open(pcm_eps_txt, "r") as f:
        lines = f.readlines()
    eps_dict = {}
    for line in lines:
        solvent, eps = line.split(": ")
        eps_dict[solvent.strip().lower()] = float(eps.strip())
    return eps_dict


def build_mf(mol: gto.Mole, args: argparse.Namespace):
    # convert memory from GB to MB
    max_memory = args.max_memory * 1024 if args.max_memory else None
    mol.build(basis=args.basis, charge=args.charge, spin=args.spin, max_memory=max_memory)
    mf = dft.KS(mol, xc=args.xc)

    # set solvation model
    pcm_models = {"C-PCM", "IEF-PCM", "SS(V)PE", "COSMO"}
    if args.solvation in pcm_models:
        eps_dict = read_pcm_eps()
        mf = mf.PCM()
        mf.with_solvent.method = args.solvation
        if isinstance(args.solvent, str):
            assert args.solvent.lower() in eps_dict, \
                f"Solvent {args.solvent} not found in pcm_eps.txt"
            eps = eps_dict[args.solvent.lower()]
        elif isinstance(args.solvent, (int, float)):
            eps = float(args.solvent)
        mf.with_solvent.eps = eps
    elif args.solvation == "SMD":
        mf = mf.SMD()
        mf.with_solvent.solvent = args.solvent
    
    # set other parameters
    mf.disp = args.disp
    mf.conv_tol = args.scf_conv
    mf.grids.grid_level = args.grid
    mf.max_cycle = args.scf_max_cycle
    return mf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyzfile", help="Input xyz file")
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
        "--solvation", type=str, default=None,
        help="Type of solvation model (default None)",
    )
    parser.add_argument(
        "--solvent", default=None,
        help="Name or dielectric constant of the solvent (default None)",
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
    mf = build_mf(mol, args)

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
        write_xyz(mf.mol, optfile, args.charge, args.spin)

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

