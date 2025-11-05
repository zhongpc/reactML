import argparse

import numpy as np
import ase.io
import yaml
from pyscf import symm
from ase.mep import NEB
from ase.optimize import FIRE
from ase import units

from reactML.common.utils import build_method, build_3c_method
from reactML.common.ase_interface import PySCFCalculator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="pyscf_config.yaml",
        help="Path to the configuration file",
    )
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config: dict = yaml.safe_load(f)
    
    # setup files
    inputfile: str = config.get("inputfile", "mol.xyz")
    filename = inputfile.rsplit(".", 1)[0]

    # set symmetry tolerance (hardcoded in Angstrom)
    if "symm_geom_tol" in config:
        symm.geom.TOLERANCE = config["symm_geom_tol"] / units.Bohr

    # read the initial and final geometries
    atoms_list = ase.io.read(inputfile, index=":")
    assert len(atoms_list) == 2, "Input file must contain exactly two structures: initial and final geometries."
    init_atoms, final_atoms = atoms_list[0], atoms_list[-1]
    assert np.all(init_atoms.symbols == final_atoms.symbols), \
        "Initial and final geometries must have the same atoms."

    # build method
    # replace inputfile
    input_atoms_list = [(ele, coord) for ele, coord in zip(init_atoms.symbols, init_atoms.positions)]
    config["inputfile"] = input_atoms_list  # fake inputfile but input list
    if "xc" in config and config["xc"].endswith("3c"):
        xc_3c = config["xc"]
        mf = build_3c_method(config)
    else:
        xc_3c = None
        mf = build_method(config)

    # create images
    images = [init_atoms]
    num_images = config.get("num_images", 5)
    for _ in range(num_images):
        images.append(init_atoms.copy())
    images.append(final_atoms)

    # set up the NEB
    neb = NEB(
        images=images,
        climb=config.get("ci_neb", False),
        method=config.get("neb_method", "aseneb"),
    )
    neb.interpolate(config.get("interpolate_method", "linear"))

    # set up calculators
    for i, image in enumerate(neb.images):
        if i == 0:
            image.calc = PySCFCalculator(mf, xc_3c=xc_3c)
        else:
            image.calc = PySCFCalculator(mf, xc_3c=xc_3c)

    # set up the optimizer
    trajectory = config.get("trajectory", f"{filename}_neb.traj")
    opt = FIRE(
        neb,
        trajectory=trajectory,
    )
    fmax = config.get("fmax", 0.05)  # in eV/Angstrom
    steps = config.get("neb_max_steps", 1000)
    opt.run(fmax=fmax, steps=steps)

    # save the final images
    images_filename = f"{filename}_neb.xyz"
    ase.io.write(images_filename, neb.images, format="extxyz")

    # judge if there is a potential transition state
    energies = np.array([image.get_potential_energy() for image in neb.images])
    max_energy_index = np.argmax(energies)
    if 0 < max_energy_index < len(neb.images) - 1:
        print(f"Potential transition state found at image {max_energy_index} with energy {energies[max_energy_index]:.6f} eV.")
        ase.io.write(f"{filename}_ts.xyz", neb.images[max_energy_index])
    else:
        print("There might be no transition state along the NEB path.")

if __name__ == "__main__":
    main()
