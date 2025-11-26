import os
import argparse

import h5py
import ase.io
from monty.serialization import loadfn, dumpfn

from reactML.crn_utils.json_data import MoleculeRecord
from reactML.crn_utils.fmt_convert import convert


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--input", "-i", type=str, required=True,
        help="Input file, can be smiles xyz"
    )
    parse.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output json file"
    )
    parse.add_argument(
        "--overwrite", "-w", action="store_true",
        help="Whether to overwrite existing output json file",
    )
    parse.add_argument(
        "--note", type=str, default=None,
        help="Optional note to include in the output json file"
    )
    parse.add_argument(
        "--h5", type=str, default=None,
        help="Computational results h5 file to link in the output json file"
    )
    args = parse.parse_args()

    if os.path.isfile(args.input):
        atoms = ase.io.read(args.input)
    else:
        raise NotImplementedError("Only file input is supported in this script.")
    
    xyzstring = f"{len(atoms)}\n\n"
    for atom in atoms:
        xyzstring += f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n"
    
    smiles = convert(in_data=xyzstring, infmt="xyz", outfmt="smiles")
    inchikey = convert(in_data=smiles, infmt="smiles", outfmt="inchikey")

    if args.output is None:
        output_filename = f"{inchikey}.json"
    else:
        output_filename = args.output

    # load output file if it exists
    if os.path.exists(output_filename) and not args.overwrite:
        json_data = loadfn(output_filename)
        mol_record = MoleculeRecord.from_dict(json_data)
        if mol_record.inchikey != inchikey:
            raise ValueError("Inchikey in existing json does not match input molecule.")
        mol_record.coords = atoms.get_positions().tolist()
    else:
        charge = atoms.info["charge"]
        multiplicity = atoms.info["multiplicity"]
        mol_record = MoleculeRecord(
            inchikey=inchikey,
            smiles=smiles,
            coords=atoms.get_positions().tolist(),
            charge=int(charge),
            multiplicity=int(multiplicity),
        )
    if args.note is not None:
        mol_record.note = args.note

    if args.h5 is not None:
        with h5py.File(args.h5, "r") as h5file:
            pass