import os
import argparse

import ase.io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trajfile", type=str,
        help="Input trajectory file in ASE format",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output file name for the XYZ format",
    )
    args = parser.parse_args()

    # Read the trajectory file
    atoms_traj = ase.io.Trajectory(args.trajfile, mode="r")
    
    output = args.output if args.output else os.path.splitext(atoms_traj.filename)[0] + ".xyz"
    for i, atoms in enumerate(atoms_traj):
        append = i > 0
        if "forces" in atoms.arrays or "forces" in atoms.calc.results:
            print(atoms.arrays)
            print(atoms.calc.results)
            columns = ['symbols', 'positions', 'forces']
        else:
            columns = ['symbols', 'positions']
        ase.io.write(output, atoms, format="extxyz", append=append, columns=columns)


if __name__ == "__main__":
    main()