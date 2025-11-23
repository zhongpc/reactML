import argparse

import numpy as np
import ase.io
import yaml
from pyscf import symm
from ase.mep import NEB, AutoNEB
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

    # create images
    read_images: str = config.get("read_images", None)
    if read_images is not None:  # read images
        images = ase.io.read(read_images, index=":")
        init_atoms = images[0]
        filename = read_images.rsplit(".", 1)[0]
    else:  # interpolate images
        # read the initial and final geometries
            # setup files
        inputfile: str = config.get("inputfile", "mol.xyz")
        filename = inputfile.rsplit(".", 1)[0]
        atoms_list = ase.io.read(inputfile, index=":")
        assert len(atoms_list) == 2, "Input file must contain exactly two structures: initial and final geometries."
        init_atoms, final_atoms = atoms_list[0], atoms_list[-1]
        assert np.all(init_atoms.symbols == final_atoms.symbols), \
            "Initial and final geometries must have the same atoms."
        images = [init_atoms]
        num_images = config.get("num_images", 5)
        for _ in range(num_images):
            images.append(init_atoms.copy())
        images.append(final_atoms)

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

    # optimize the terminal images first
    fmax = float(config.get("fmax", 0.05))  # in eV/Angstrom
    use_ci_neb = config.get("ci_neb", False)
    max_steps = int(config.get("max_steps", 1000))
    use_soscf: bool = config.get("soscf", False)
    spring_constant: float = config.get("spring_constant", 0.1)
    method: str = config.get("neb_method", "improvedtangent")
    max_unconverged_steps: int = config.get("max_unconverged_steps", None)
    opt_terminal: bool = config.get("optimize_terminal", True)
    interpolate_method: str = config.get("interpolate_method", "idpp")
    if read_images is None and opt_terminal:
        images[0].calc = PySCFCalculator(
            mf, xc_3c=xc_3c, soscf=use_soscf, max_unconverged_steps=max_unconverged_steps
        )
        images[-1].calc = PySCFCalculator(
            mf, xc_3c=xc_3c, soscf=use_soscf, max_unconverged_steps=max_unconverged_steps
        )
        print("Optimizing the initial image.")
        with FIRE(images[0]) as opt0:
            opt0.run(fmax=fmax)
        print("Optimizing the final image.")
        with FIRE(images[-1]) as opt1:
            opt1.run(fmax=fmax)
        del opt0, opt1

    # set up the NEB
    ci_after_n_steps: int = config.get("ci_after_n_steps", None)
    assert (ci_after_n_steps is None) ^ use_ci_neb, \
        "If turning on CI-NEB, please specify 'ci_after_n_steps' in the config file.\n\
            If setting 'ci_after_n_steps', please also turn on CI-NEB."
    if ci_after_n_steps is not None:
        print(f"Climbing-image NEB will start after {ci_after_n_steps} steps.")

    neb = NEB(
        images=images,
        k=spring_constant,
        climb=False,
        remove_rotation_and_translation=True,
        method=method,
    )
    if read_images is None:
        neb.interpolate(interpolate_method)

    # set up calculators
    for i, image in enumerate(images):
        if image.calc is not None:
            continue  # already has a calculator (e.g., terminal images)
        image.calc = PySCFCalculator(
            mf, xc_3c=xc_3c, soscf=use_soscf, max_unconverged_steps=max_unconverged_steps
        )

    # set up the optimizer
    trajectory = config.get("trajectory", f"{filename}_neb.traj")
    opt = FIRE(
        neb,
        trajectory=trajectory,
    )
    max_steps = config.get("neb_max_steps", 1000)
    steps = max_steps if ci_after_n_steps is None else ci_after_n_steps
    opt.run(fmax=fmax, steps=steps)
    if ci_after_n_steps is not None:
        ci_neb = NEB(
            images=images,
            k=spring_constant,
            climb=True,
            remove_rotation_and_translation=True,
            method=method,
        )
        opt = FIRE(
            ci_neb,
            trajectory=trajectory,
            append_trajectory=True,
        )
        opt.run(fmax=fmax, steps=max_steps-ci_after_n_steps)
        images = ci_neb.images
    else:
        images = neb.images

    # save the final images
    images_filename = f"{filename}_neb.xyz"
    ase.io.write(images_filename, images, format="extxyz")
    # judge if there is a potential transition state
    energies = np.array([image.get_potential_energy() for image in images])
    for i, energy in enumerate(energies):
        print(f"Image {i}: Energy = {energy:.6f} eV")
    # the highest energy point is the potential transition state
    max_energy_index = np.argmax(energies)
    if 0 < max_energy_index < len(images) - 1:
        print(f"A potential transition state is found at image {max_energy_index}.")
        ase.io.write(f"{filename}_ts.xyz", images[max_energy_index])
    else:
        print("There might be no transition state along the NEB path.")


if __name__ == "__main__":
    main()
