import os
import argparse
from types import MethodType
from copy import copy

import numpy as np
import ase.io
import yaml
from pyscf import symm
from ase.mep import AutoNEB, NEB
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

    # set symmetry tolerance (hardcoded in Angstrom)
    if "symm_geom_tol" in config:
        symm.geom.TOLERANCE = config["symm_geom_tol"] / units.Bohr

    # read the initial and final geometries
    inputfile: str = config.get("inputfile", "mol.xyz")
    filename = inputfile.rsplit(".", 1)[0]
    atoms_list = ase.io.read(inputfile, index=":")
    assert len(atoms_list) == 2, "Input file must contain exactly two structures: initial and final geometries."
    init_atoms, final_atoms = atoms_list[0], atoms_list[-1]
    assert np.all(init_atoms.symbols == final_atoms.symbols), \
        "Initial and final geometries must have the same atoms."

    # build method
    # replace inputfile
    charge = config.get("charge", 0)
    multiplicity = config.get("spin", 0) + 1  # PySCF uses 2S, MLIP uses 2S+1
    input_atoms_list = [(ele, coord) for ele, coord in zip(init_atoms.symbols, init_atoms.positions)]
    config["inputfile"] = input_atoms_list  # fake inputfile but input list
    if "mlip" in config:
        assert "xc" not in config, "Please do not specify 'xc' when using MLIP."
        if config["mlip"] == "uma":
            from fairchem.core import FAIRChemCalculator, pretrained_mlip
            from omegaconf import OmegaConf
            device: str = config.get("device", None)
            model: str = config["model"]
            task_name: str = config.get("task_name", "omol")
            atom_refs = OmegaConf.load(os.path.join(model.rsplit('/', 1)[0], "iso_atom_elem_refs.yaml"))
            predictor = pretrained_mlip.load_predict_unit(model, device=device, atom_refs=atom_refs)
            calc = FAIRChemCalculator(predictor, task_name=task_name)
        else:
            raise NotImplementedError(f"Unknown MLIP: {config['mlip']}")
    elif "xc" in config:
        assert "mlip" not in config, "Please do not specify 'mlip' when using DFT."
        if config["xc"].endswith("3c"):
            xc_3c = config["xc"]
            mf = build_3c_method(config)
        else:
            xc_3c = None
            mf = build_method(config)
        use_soscf: bool = config.get("soscf", False)
        max_unconverged_steps: int = config.get("max_unconverged_steps", None)
        calc = PySCFCalculator(
            method=mf, xc_3c=xc_3c, soscf=use_soscf,
            max_unconverged_steps=max_unconverged_steps
        )
    else:
        raise ValueError("Please specify either 'mlip' or 'xc' in the config file.")

    max_images: int = config.get("max_images", 11)
    fmax = float(config.get("fmax", 0.05))  # in eV/Angstrom
    use_climb = config.get("climb", False)
    max_steps = int(config.get("max_steps", 1000))
    spring_constant: float = float(config.get("spring_constant", 0.1))
    method: str = config.get("neb_method", "improvedtangent")
    opt_terminal: bool = config.get("optimize_terminal", True)
    interpolate_method: str = config.get("interpolate_method", "idpp")

    def attach_calculator(images):
        for image in images:
            image.info["charge"] = charge
            image.info["spin"] = multiplicity  # MLIP uses 2S+1, PySCF uses 2S
            if not isinstance(image.calc, type(calc)):
                image.calc = copy(calc)
    
    regenerate_terminal: bool = config.get("regenerate_terminal", False)
    if regenerate_terminal:
        images = [init_atoms]
        for _ in range(max_images):
            images.append(init_atoms.copy())
        images.append(final_atoms)
        attach_calculator(images)
        neb = NEB(images=images, method=method, allow_shared_calculator=True)
        neb.interpolate(interpolate_method)
        energies = [image.get_potential_energy() for image in images]
        max_energy_index = energies.index(max(energies))
        init_atoms.set_positions(images[max_energy_index - 1].positions)
        final_atoms.set_positions(images[max_energy_index + 1].positions)

    # optimize the terminal images first
    if opt_terminal or regenerate_terminal:
        attach_calculator([init_atoms, final_atoms])
        print("Optimizing the initial image.")
        with FIRE(init_atoms) as opt0:
            opt0.run(fmax=fmax)
        print("Optimizing the final image.")
        with FIRE(final_atoms) as opt1:
            opt1.run(fmax=fmax)
        del opt0, opt1

    # write the initial and final to prefix_000.traj and prefix_001.traj
    ase.io.write(f"{filename}_000.traj", init_atoms)
    ase.io.write(f"{filename}_001.traj", final_atoms)

    def patched_get_energies(self: AutoNEB):
        energies = []
        for a in self.all_images:
            if not isinstance(a.calc, type(calc)):
                a.calc = copy(calc)
            energies.append(a.get_potential_energy())
        return np.array(energies)

    autoneb = AutoNEB(
        attach_calculators=attach_calculator,
        iter_folder=f"{filename}_iter",
        prefix=f"{filename}_",
        n_simul=1,
        n_max=max_images,
        climb=use_climb,
        fmax=fmax,
        maxsteps=max_steps,
        k=spring_constant,
        method=method,
        optimizer=FIRE,
        interpolate_method=interpolate_method,
        parallel=False,
    )
    autoneb.get_energies = MethodType(patched_get_energies, autoneb)
    images = autoneb.run()

    # save the final images
    images_filename = f"{filename}_neb.xyz"
    ase.io.write(images_filename, images, format="extxyz")
    # judge if there is a potential transition state
    energies = np.array([image.get_potential_energy() for image in images])
    for i, energy in enumerate(energies):
        print(f"Image {i:02d}: Energy = {energy:.6f} eV")
    # the highest energy point is the potential transition state
    max_energy_index = np.argmax(energies)
    if 0 < max_energy_index < len(images) - 1:
        print(f"A potential transition state is found at image {max_energy_index}.")
        ase.io.write(f"{filename}_ts.xyz", images[max_energy_index])
    else:
        print("There might be no transition state along the NEB path.")


if __name__ == "__main__":
    main()