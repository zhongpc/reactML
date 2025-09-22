import argparse

import h5py
import ase.io


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--xyz", type=str, required=True, help="Input XYZ file")
    argparser.add_argument("--h5", type=str, required=True, help="Input HDF5 file")
    argparser.add_argument("--out", type=str, default=None, help="Output HDF5 file")
    args = argparser.parse_args()

    atoms = ase.io.read(args.xyz)
    with h5py.File(args.h5, "r") as f:
        freq_wavenumbers = f["freq_wavenumber"][:]
        norm_mode = f["norm_mode"][:]
    
    n_atoms = len(atoms)
    n_modes = len(freq_wavenumbers)

    if args.out is None:
        outputfile = args.h5.rsplit(".", 1)[0] + "_vib.txt"
    else:
        outputfile = args.out
    
    with open(outputfile, "w") as f:
        f.write(f"{n_atoms} {n_modes}\n\n")
        for i, (freq, mode) in enumerate(zip(freq_wavenumbers, norm_mode)):
            f.write(f"N {freq:.4f} NULL {i+1}\n")
            for m in mode.reshape(-1):
                f.write(f"{m:.4f}\n")
            if i < n_modes - 1:
                f.write("\n")
            else:
                f.write("END\n")


if __name__ == "__main__":
    main()
