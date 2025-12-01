import os
import argparse

import h5py
import yaml
import ase.io
from monty.serialization import dumpfn

from reactML.crn_utils.json_data import MoleculeRecord
from reactML.crn_utils.fmt_convert import convert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file (not used in this script)."
    )
    parser.add_argument(
        "--smiles", type=str, default=None, help="SMILES string of the molecule (not used in this script)."
    )
    parser.add_argument(
        "--backend", type=str, default=None, help="Backend to use: 'rdkit' or 'openbabel'."
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config: dict = yaml.safe_load(f)

    # determine level of theory string
    xc: str = config["xc"]
    if xc.lower().endswith("3c"):
        level_of_theory = xc
    else:
        basis: str = config["basis"]
        level_of_theory = f"{xc}/{basis}"
    with_solvent: bool = config.get("with_solvent", False)
    if with_solvent:
        solv_method: str = config["solvent"]["method"]
        if "solvent" in config["solvent"]:
            solvent: str = config["solvent"]["solvent"]
            level_of_theory += f"+{solv_method}({solvent})"
        elif "eps" in config["solvent"]:
            eps: float = config["solvent"]["eps"]
            level_of_theory += f"/{solv_method}(eps={eps})"
        else:
            level_of_theory += f"/{solv_method}"

    # load molecule from xyz
    inputfile: str = config["inputfile"]
    filename: str = inputfile.rsplit('.', 1)[0]
    datafile: str = config.get("datafile", f"{filename}_data.h5")
    # if opt in datafile name, replace with geom opt
    ran_opt: bool = config.get("opt", False)
    optfile = f"{filename}_opt.xyz"
    if ran_opt and os.path.exists(optfile):
        inputfile = optfile
    atoms = ase.io.read(inputfile)

    # determine the charge and multiplicity
    if "charge" in config:
        charge = config["charge"]
    elif "charge" in atoms.info:
        charge = atoms.info["charge"]
    else:
        raise ValueError("Charge must be specified in the configuration file or in the input XYZ file.")
    if "spin" in config:
        multiplicity = config["spin"] + 1
    elif "multiplicity" in atoms.info:
        multiplicity = atoms.info["multiplicity"]
    else:
        raise ValueError("Multiplicity must be specified in the configuration file or in the input XYZ file.")
    
    # generate inchikey and smiles
    if args.smiles is None:
        xyzstring = f"{len(atoms)}\n\n"
        for atom in atoms:
            xyzstring += f"{atom.symbol} {atom.position[0]} {atom.position[1]} {atom.position[2]}\n"
        smiles = convert(in_data=xyzstring, infmt="xyz", outfmt="smiles", backend=args.backend, charge=charge)
    else:
        smiles = args.smiles
    inchikey = convert(in_data=smiles, infmt="smiles", outfmt="inchikey", backend=args.backend)

    # read the potential json file
    json_file = f"{filename}.json"
    # if exists, make a copy
    if os.path.exists(json_file):
        os.rename(json_file, f"{json_file}.bak")

    # create MoleculeRecord
    mol_record = MoleculeRecord(
        elements=[atom.symbol for atom in atoms],
        coords=atoms.get_positions(),
        smiles=smiles,
        inchikey=inchikey,
        charge=charge,
        multiplicity=multiplicity,
        note=f"Created from {inputfile} at level of theory {level_of_theory}",
    )
    # add method results from datafile if exists
    if os.path.exists(datafile):
        results = {}
        possible_keys = [
            "T", "P", "E0", "ZPE", "U", "H", "S", "G",
        ]
        with h5py.File(datafile, 'r') as h5f:
            for key in possible_keys:
                if key not in h5f:
                    continue
                unit = h5f[f"{key}_unit"][()].decode()
                value = h5f[f"{key}"][()]
                results[f"{key}/{unit}"] = float(value)
        mol_record.add_results(level_of_theory=level_of_theory, results=results)
    
    # dump to json
    dumpfn(mol_record.to_dict(), json_file, indent=2)


if __name__ == "__main__":
    main()