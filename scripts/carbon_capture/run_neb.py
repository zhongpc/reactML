import argparse
import ase.io
from ase.mep import NEB

from pyscf import gto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "initxyz", type=str,
        help="Initial geometry in xyz format",
    )
    parser.add_argument(
        "finalxyz", type=str,
        help="Final geometry in xyz format",
    )
    parser.add_argument(
        "--prefix", "-p", type=str, default="neb",
        help="Prefix for output files",
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
    args = parser.parse_args()

    # read the initial and final geometries
    init_mol = gto.Mole.fromfile(filename=args.initxyz)
    final_mol = gto.Mole.fromfile(filename=args.finalxyz)
    init_atoms = ase.io.read(args.initxyz, format="xyz")
    final_atoms = ase.io.read(args.finalxyz, format="xyz")

    # create images
    images = [init_atoms]
    for _ in range(args.num_images):
        images.append(init_atoms.copy())
    images.append(final_atoms)

    # set up the NEB
    neb = NEB(
        images=images,
        
    )
    neb.interpolate(method=args.interpolate_method)
    
        
    # auto_neb = AutoNEB(
        


if __name__ == "__main__":
    main()
    