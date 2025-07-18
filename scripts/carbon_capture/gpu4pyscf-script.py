import argparse
from pyscf import gto
from gpu4pyscf import dft
from pyscf.geomopt import geometric_solver
from pyscf.hessian import thermo


def write_xyz(mol: gto.Mole, filename: str, charge: int = 0, spin: int = 1):
    elements = mol.elements
    coords = mol.atom_coords(unit="Angstrom")
    with open(filename, "w") as f:
        f.write(f"{len(elements)}\n")
        f.write(f"{charge} {spin + 1}\n")
        for ele, coord in zip(elements, coords):
            f.write(f"{ele: <2}    {coord[0]:10.6f}    {coord[1]:10.6f}    {coord[2]:10.6f}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xyzfile", help="Input xyz file")
    parser.add_argument("--xc", "-f", type=str, default="B3LYP", help="Name of Exchange-Correlation Functional")
    parser.add_argument("--basis", "-b", type=str, default="6-31++G(d,p)", help="Name of Basis Set")
    parser.add_argument("--charge", "-c", type=int, default=0, help="Total charge")
    parser.add_argument("--spin", "-s", type=int, default=0, help="Total spin (2S not 2S+1)")
    parser.add_argument("--disp", "-d", type=str, default=None, help="Type of Dispersion Correction")
    parser.add_argument("--temp", "-T", type=float, default=298.15, help="Temperature for thermodynamic analysis")
    parser.add_argument("--press", "-p", type=float, default=101325., help="Pressure for thermodynamic analysis")
    parser.add_argument("--opt", "-o", action="store_true", help="Whether to do optimize the geometry")
    parser.add_argument("--freq", action="store_true", help="Whether to calculate the frequencies")
    parser.add_argument("--max-memory", type=int, default=10000, help="Maximum memory in MB")
    args = parser.parse_args()

    # read the xyz file with pyscf
    mol = gto.Mole()
    mol.fromfile(filename=args.xyzfile)

    # build the molecule with C-PCM solvation model
    mol.build(basis=args.basis, charge=args.charge, spin=args.spin, max_memory=args.max_memory)
    mf = dft.KS(mol, xc=args.xc).PCM()
    mf.disp = args.disp
    mf.max_cycle = 200
    mf.with_solvent.method = "C-PCM"
    mf.with_solvent.eps = 46.826  # DMSO

    # geometric optimization
    if args.opt:
        conv, mol = geometric_solver.kernel(mf)
        # save optimized struct
        optfile = args.xyzfile.replace(".xyz", "_opt.xyz")
        if not conv:
            print("Geometry Optimization not converged!!")
            return
        write_xyz(mol, optfile, args.charge, args.spin)
        mol.build(basis=args.basis, charge=args.charge, spin=args.spin, max_memory=args.max_memory)
    
    # single point calculation
    mf = dft.KS(mol, xc=args.xc).PCM()
    mf.disp = args.disp
    mf.max_cycle = 200
    mf.with_solvent.method = "C-PCM"
    mf.with_solvent.eps = 46.826  # DMSO
    mf.run()

    # hessian calculation
    if args.freq:
        hessian = mf.Hessian().kernel()
        freq_info = thermo.harmonic_analysis(mol, hessian)
        thermo_info = thermo.thermo(mf, freq_info["freq_au"], args.temp, args.press)
        thermo.dump_normal_mode(mol, freq_info)
        thermo.dump_thermo(mol, thermo_info)

        # exclude translation contributions
        G_tot_exclude_trans = thermo_info["G_tot"][0] - thermo_info["G_trans"][0]
        print("Gibbs free energy without translation contributions [Eh]", f"{G_tot_exclude_trans:.5f}")


if __name__ == "__main__":
    main()

