import argparse

import numpy as np
from pyscf import gto
import ase.io
from ase.mep import NEB
from ase.optimize import FIRE
from ase import units

from reactML.common.utils import build_dft
from reactML.common.ase_interface import PySCFCalculator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xyzfile", type=str,
        help="Geometry file in XYZ format containing initial and final structures",
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
        "--max-memory", type=int, default=None,
        help="Maximum memory in GB",
    )
    parser.add_argument(
        "--num-images", type=int, default=5,
        help="Number of images in NEB (default 5)",
    )
    parser.add_argument(
        "--spring-constant", "-k", type=float, default=0.1,
        help="Spring constant for NEB (default 0.1)",
    )
    parser.add_argument(
        "--climbing-image", "-ci", action="store_true",
        help="Whether to use CI-NEB (default: False)",
    )
    parser.add_argument(
        "--neb-method", type=str,
        choices=["aseneb", "eb", "improvedtangent", "spline", "string"],
        default="aseneb",
        help="Method for NEB (default: aseneb)",
    )
    parser.add_argument(
        "--interpolate-method", type=str,
        choices=["linear", "idpp"], default="linear",
        help="Interpolation method for NEB (default: linear)",
    )
    parser.add_argument(
        "--opt-fmax", type=float, default=4.5e-4,
        help="Maximum force for optimization convergence (default 4.5e-4 a.u.)",
    )
    parser.add_argument(
        "--opt-max-steps", type=int, default=100_000_000,
        help="Maximum number of optimization steps (default 100,000,000)",
    )
    args = parser.parse_args()

    # read the initial and final geometries
    atoms_list = ase.io.read(args.xyzfile, format="xyz", index=":")
    assert len(atoms_list) == 2, "Input file must contain exactly two structures: initial and final geometries."
    init_atoms, final_atoms = atoms_list[0], atoms_list[-1]
    assert np.all(init_atoms.symbols == final_atoms.symbols), \
        "Initial and final geometries must have the same atoms."

    # pyscf will only read the first structure from the file
    mol = gto.Mole(charge=args.charge, spin=args.spin)  # mol.build() will be called in mol.fromfile
    mol.fromfile(filename=args.xyzfile)                 # we must set charge and spin before reading the file
    
    # create images
    images = [init_atoms]
    for _ in range(args.num_images):
        images.append(init_atoms.copy())
    images.append(final_atoms)

    filename = args.xyzfile.split(".", -1)[0]

    # set up the NEB
    neb = NEB(
        images=images,
        climb=args.climbing_image,
        method=args.neb_method,
    )
    neb.interpolate(method=args.interpolate_method)
    
    # set up calculators
    mf = build_dft(mol, **vars(args))
    for i, image in enumerate(neb.images):
        if i == 0:
            image.calc = PySCFCalculator(method=mf)
        else:
            image.calc = PySCFCalculator(method=mf.copy())
    
    # set up the optimizer
    opt = FIRE(
        neb,
        trajectory=f"{filename}_neb.traj",
    )
    fmax = args.opt_fmax * units.Hartree / units.Bohr  # Convert from Hartree/Bohr
    opt.run(fmax=fmax, steps=args.opt_max_steps)
        

if __name__ == "__main__":
    main()
    