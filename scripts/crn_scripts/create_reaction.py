import os
import argparse

import h5py
import yaml
import ase.io
from monty.serialization import dumpfn

from reactML.common.utils import build_method, build_3c_method
from reactML.crn_utils.json_data import ReactionRecord
from reactML.crn_utils.fmt_convert import convert


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default=None, help="Path to config file."
    )
    parser.add_argument(
        "--reactants", type=str, nargs='+', default=None, help="List of reactant SMILES strings."
    )
    parser.add_argument(
        "--products", type=str, nargs='+', default=None, help="List of product SMILES strings."
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
        level_of_theory = build_3c_method(xc)
    else:
        basis: str = config["basis"]
        level_of_theory = build_method(xc, basis)
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

    # create reaction record
    reaction_record = ReactionRecord(
        reactants=args.reactants,
        products=args.products,
        level_of_theory=level_of_theory
    )

    # save reaction record to json
    outputfile: str = config.get("outputfile", "reaction_record.json")
    dumpfn(reaction_record.dict(), outputfile)
    print(f"Reaction record saved to {outputfile}")