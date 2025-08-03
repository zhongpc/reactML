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
        if i == 0:
            # Write the first frame to the output file
            ase.io.write(output, atoms, format="extxyz", append=False)
        else:  # Append subsequent frames
            ase.io.write(output, atoms, format="extxyz", append=True)


if __name__ == "__main__":
    main()