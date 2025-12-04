import os
import argparse

import h5py
import yaml
import ase.io
from ase.units import Hartree, kcal, mol, kJ
from monty.serialization import dumpfn

from reactML.crn_utils.json_data import ReactionRecord
from reactML.crn_utils.fmt_convert import convert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, default=None, help="Path to config file."
    )
    parser.add_argument(
        "--filename", "-f", type=str, default=None, help="Base filename for input/output."
    )
    parser.add_argument(
        "--note", "-n", type=str, default="", help="Note to add to the reaction record."
    )
    parser.add_argument(
        "--backend", "-b", type=str, default="openbabel", help="Backend for format conversion."
    )
    parser.add_argument(
        "--energy-threshold", "-e", type=float, default=None, help="Free energy threshold (in eV) to filter reactions."
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
    filename: str = args.filename
    react_xyzfile = f"{filename}_r_opt.xyz"
    react_datafile = f"{filename}_r_data.h5"
    prod_xyzfile = f"{filename}_p_opt.xyz"
    prod_datafile = f"{filename}_p_data.h5"
    react_atoms = ase.io.read(react_xyzfile)
    prod_atoms = ase.io.read(prod_xyzfile)

    # determine the charge and multiplicity
    if "charge" in config:
        charge: int = config["charge"]
    elif "charge" in react_atoms.info:
        charge = react_atoms.info["charge"]
    elif "charge" in prod_atoms.info:
        charge = prod_atoms.info["charge"]
    else:
        raise ValueError("Charge not found in config or atoms info.")
    if "spin" in config:
        multiplicity: int = config["spin"] + 1
    elif "multiplicity" in react_atoms.info:
        multiplicity = react_atoms.info["multiplicity"]
    elif "multiplicity" in prod_atoms.info:
        multiplicity = prod_atoms.info["multiplicity"]
    else:
        raise ValueError("Multiplicity not found in config or atoms info.")
    
    # convert xyz to an entire smiles
    backend: str = args.backend
    with open(react_xyzfile, 'r') as f:
        lines = f.readlines()
        # empty the second line (comment line)
        lines[1] = "\n"
        react_xyzstr = ''.join(lines)
    react_smiles: str = convert(react_xyzstr, infmt='xyz', outfmt='smiles', backend=backend, charge=charge, multiplicity=multiplicity)
    with open(prod_xyzfile, 'r') as f:
        lines = f.readlines()
        # empty the second line (comment line)
        lines[1] = "\n"
        prod_xyzstr = ''.join(lines)
    prod_smiles: str = convert(prod_xyzstr, infmt='xyz', outfmt='smiles', backend=backend, charge=charge, multiplicity=multiplicity)

    # separate smiles
    print(f"Reactant SMILES: {react_smiles}")
    print(f"Product SMILES: {prod_smiles}")
    # create reaction record
    try:
        reaction_record = ReactionRecord(
            elements=react_atoms.get_chemical_symbols(),
            reactant_coords=react_atoms.get_positions().tolist(),
            product_coords=prod_atoms.get_positions().tolist(),
            reactant_smiles=react_smiles,
            product_smiles=prod_smiles,
            charge=charge,
            multiplicity=multiplicity,
            note=args.note if args.note else f"Created from {react_xyzfile} and {prod_xyzfile} at level of theory {level_of_theory}",
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create ReactionRecord for {filename}: {e}")
    if reaction_record.reactant_inchikeys == reaction_record.product_inchikeys:
        print(f"Warning: Reactant and product inchikeys are the same in {filename}.")
        exit(1)
    
    # read the results from datafiles if exist
    possible_keys = [
        "T", "P", "E0", "ZPE", "U", "H", "S", "G",
    ]
    thermo_keys = {"E0", "U", "H", "S", "G"}
    react_results = {}
    prod_results = {}
    rxn_results = {}
    Delta_G: float = None
    Delta_G_unit: str = None
    if os.path.exists(react_datafile):
        with h5py.File(react_datafile, 'r') as h5f:
            for key in possible_keys:
                if key not in h5f:
                    continue
                unit = h5f[f"{key}_unit"][()].decode()
                value = h5f[f"{key}"][()]
                react_results[f"{key}/{unit}"] = float(value)
    if os.path.exists(prod_datafile):
        with h5py.File(prod_datafile, 'r') as h5f:
            for key in possible_keys:
                if key not in h5f:
                    continue
                unit = h5f[f"{key}_unit"][()].decode()
                key_unit = f"{key}/{unit}"
                value = h5f[f"{key}"][()]
                prod_results[key_unit] = float(value)
                if key in thermo_keys and key_unit in react_results:
                    rxn_results[f"Delta_{key_unit}"] = float(value) - float(react_results[key_unit])
                if key == "G":
                    Delta_G = rxn_results[f"Delta_{key_unit}"]
                    Delta_G_unit = unit

    if args.energy_threshold is not None:
        assert Delta_G is not None, "Gibbs free energy change not found; cannot apply energy threshold."
        if Delta_G_unit.lower() == "ev":
            pass
        elif Delta_G_unit.lower() in {"eh", "hartree", "a.u."}:
            Delta_G *= Hartree
        elif Delta_G_unit.lower() in {"kcal/mol"}:
            Delta_G *= kcal / mol
        elif Delta_G_unit.lower() in {"kj/mol"}:
            Delta_G *= kJ / mol
        else:
            raise NotImplementedError(f"Unit conversion for {Delta_G_unit} not implemented.")
        if Delta_G > args.energy_threshold:
            print(f"Reaction filtered out due to energy threshold: ΔG = {Delta_G:.4f} eV > {args.energy_threshold:.4f} eV")
            exit(0)

    reaction_record.add_results(
        level_of_theory=level_of_theory,
        reactant_results=react_results,
        product_results=prod_results,
        reaction_results=rxn_results,
    )
    # dump to json
    json_file = f"{filename}.json"
    if os.path.exists(json_file):
        os.rename(json_file, f"{json_file}.bak")
    dumpfn(reaction_record.to_dict(), json_file, indent=2)


if __name__ == "__main__":
    main()