import argparse
import time
from typing import List

import yaml
from pymatgen.core.structure import Molecule

from reactML.crn_utils.frag_recombine import FragmentationRecombination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file."
    )
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config: dict = yaml.safe_load(file)

    # read the input molecules
    molecules: List[Molecule] = []
    frag_depths: List[int] = []
    inputfiles: List[dict] = config["inputfiles"]
    for inputfile_dict in inputfiles:
        inputfile = inputfile_dict["inputfile"]
        charge = inputfile_dict["charge"]
        multiplicity = inputfile_dict["multiplicity"]
        frag_depth = inputfile_dict["frag_depth"]
        molecule = Molecule.from_file(inputfile)
        molecule.set_charge_and_spin(charge, multiplicity)
        molecules.append(molecule)
        frag_depths.append(frag_depth)

    
    frag_recombine = FragmentationRecombination(
        molecules=molecules,
        frag_depths=frag_depths,
        bonding_scale_max=float(config.get("bonding_scale_max", 1.0)),
        bonding_scale_min=float(config.get("bonding_scale_min", 1.0)),
        n_bonding_scales=int(config.get("n_bonding_scales", 1)),
        n_angles=int(config.get("n_angles", 100)),
        early_stop_angular=config.get("early_stop_angular", True),
        localopt=config.get("localopt", False),
        output_dir=str(config.get("output_dir", "./struct")),
    )
    frag_recombine.run(add_monoatomic=False)


if __name__ == "__main__":
    main()