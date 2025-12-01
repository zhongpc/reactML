import argparse
import os

import yaml
import numpy as np
import ase.io
from ase import Atoms
from ase.data import vdw_radii
from monty.serialization import loadfn
from yamo.yamol import yamol
from reactnet.netgen.producer.gen_prods import reaction_enumeration, create_reaction_xyzs

from reactML.crn_utils.json_data import MoleculeRecord


def get_extented_box(atoms: Atoms, padding: float = 0.0) -> list:
    """Get the extended bounding box of an ASE Atoms object.

    Args:
        atoms (Atoms): The ASE Atoms object.
        padding (float, optional): Padding to add to each side of the box. Defaults to 0.0.

    Returns:
        np.ndarray: The extended bounding box as [x_min, x_max, y_min, y_max, z_min, z_max].
    """
    positions = atoms.get_positions()
    x_min = positions[:, 0].min() - padding
    x_max = positions[:, 0].max() + padding
    y_min = positions[:, 1].min() - padding
    y_max = positions[:, 1].max() + padding
    z_min = positions[:, 2].min() - padding
    z_max = positions[:, 2].max() + padding
    return np.array([[x_min, x_max], [y_min, y_max], [z_min, z_max]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file."
    )
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config: dict = yaml.safe_load(file)
    input_format: str = config.get("input_format", "xyz")
    if input_format.lower() == "xyz":
        xyz_tmp_dir: str = config.get("xyz_tmp_dir", "xyz_temp")
        xyz_tmp_dir = os.path.join(os.getcwd(), xyz_tmp_dir)
        os.makedirs(xyz_tmp_dir, exist_ok=True)
    # read the molecules in the pool
    pool_dir: str = config["pool_dir"]
    if not os.path.exists(pool_dir):
        raise FileNotFoundError(f"Molecule pool path '{pool_dir}' does not exist.")
    inputs = []
    names = []
    for json_file in os.listdir(pool_dir):
        if not json_file.endswith('.json'):
            continue
        mol_record: MoleculeRecord = loadfn(os.path.join(pool_dir, json_file))
        if input_format.lower() == "xyz":
            atoms = mol_record.to_ase_atoms()
            inputs.append(atoms)
        elif input_format.lower() == "smiles":
            inputs.append(mol_record.smiles)
        else:
            raise ValueError(f"Unsupported input format '{input_format}'.")
        names.append(json_file.rsplit('.', 1)[0])
    base_pool_dir: str = config.get("base_pool_dir", None)
    base_inputs = []
    base_names = []
    if base_pool_dir is not None:
        if not os.path.exists(base_pool_dir):
            raise FileNotFoundError(f"Base molecule pool path '{base_pool_dir}' does not exist.")
        for json_file in os.listdir(base_pool_dir):
            if not json_file.endswith('.json'):
                continue
            mol_record: MoleculeRecord = loadfn(os.path.join(base_pool_dir, json_file))
            if input_format.lower() == "xyz":
                atoms = mol_record.to_ase_atoms()
                base_inputs.append(atoms)
            elif input_format.lower() == "smiles":
                base_inputs.append(mol_record.smiles)
            else:
                raise ValueError(f"Unsupported input format '{input_format}'.")
            base_names.append(json_file.rsplit('.', 1)[0])
    
    # prepare for the tmp dirs
    xtb_opt: bool = config.get("xtb_opt", True)
    xtb_temp_dir: str = config.get("xtb_temp_dir", "xTB_temp")
    xtb_temp_dir = os.path.join(os.getcwd(), xtb_temp_dir)
    if xtb_opt:
        os.makedirs(xtb_temp_dir, exist_ok=True)
    # prepare for the reaction xyzs output dir
    reactions_dir: str = config.get("reactions_dir", "reactions")
    reactions_dir = os.path.join(os.getcwd(), reactions_dir)
    os.makedirs(reactions_dir, exist_ok=True)

    # perform reaction enumeration
    n_bonds_break: int = config.get("n_bonds_break", 2)
    n_bonds_form: int = config.get("n_bonds_form", 2)
    n_workers: int = config.get("n_workers", 8)
    rcs_filter: bool = config.get("rcs_filter", True)
    rcs_threshold: float = config.get("rcs_threshold", 0.3)
    model_path: str = config.get("model_path", None)
    xyzfiles_all = []
    if rcs_filter and model_path is None:
        raise ValueError("rcs_filter is set to True, but no model_path is provided in the config.")
    # 1. sinlge molecule reactions
    for inp, name in zip(inputs, names):
        if input_format.lower() == "xyz":
            xyzfile = os.path.join(xyz_tmp_dir, f"{name}.xyz")
            ase.io.write(xyzfile, inp)
            rinp = xyzfile
        elif input_format.lower() == "smiles":
            rinp = inp
        else:
            raise ValueError(f"Unsupported input format '{input_format}'.")
        print(f"Enumerating reactions for molecule '{name}'...")
        reactions = reaction_enumeration(
            rinp, nb_break=n_bonds_break, nb_form=n_bonds_form, message=False,
            check_bimol=False, n_workers=n_workers,
        )
        subdir = os.path.join(reactions_dir, name)
        rmol = yamol(rinp)
        os.makedirs(subdir, exist_ok=True)
        xyzfiles = create_reaction_xyzs(
            rmol, reactions, work_folder=xtb_temp_dir,
            output_folder=subdir, model_path=model_path,
            n_workers=n_workers, xtb_opt=xtb_opt,
            rcs_filter=rcs_filter, rcs_thresh=rcs_threshold,
            message=False, use_changed_geometry=True,
        )
        print(f"Generated {len(xyzfiles)} reaction xyz files for molecule '{name}'.")
        xyzfiles_all.extend(xyzfiles)
    
    # 2. bimolecular reactions
    mixed_inputs = inputs + base_inputs
    mixed_names = names + base_names
    for i in range(len(inputs)):
        for j in range(i + 1, len(mixed_inputs)):
            print(f"Enumerating reactions for molecule pair '{names[i]}' and '{mixed_names[j]}'...")
            inpA = inputs[i]
            nameA = names[i]
            inpB = mixed_inputs[j]
            nameB = mixed_names[j]
            name = f"{nameA}-{nameB}"
            if input_format.lower() == "xyz":
                # combine two molecules into one xyz file
                atomsA: Atoms = inpA
                atomsB: Atoms = inpB
                extentA = get_extented_box(atomsA)
                extentB = get_extented_box(atomsB)
                # calculate the safe distance to separate the two molecules
                safe_distance = vdw_radii[atomsA.get_atomic_numbers()].max() + \
                                vdw_radii[atomsB.get_atomic_numbers()].max() + 0.5  # in Angstrom
                # shift molecule B
                shift_vec = np.zeros(3)
                shift_vec[0] = extentA[0, 1] - extentB[0, 0] + safe_distance
                atomsB_shifted = atomsB.copy()
                atomsB_shifted.translate(shift_vec)
                # combine and write to xyz file
                combined_atoms = atomsA + atomsB_shifted
                xyzfile = os.path.join(xyz_tmp_dir, f"{name}.xyz")
                ase.io.write(xyzfile, combined_atoms)
                rinp = xyzfile
            elif input_format.lower() == "smiles":
                smilesA = inpA
                smilesB = inpB
                smiles = f"{smilesA}.{smilesB}"
                rinp = smiles
            else:
                raise ValueError(f"Unsupported input format '{input_format}'.")
            reactions = reaction_enumeration(
                rinp, nb_break=n_bonds_break, nb_form=n_bonds_form, message=False,
                check_bimol=True, n_workers=n_workers,
            )
            subdir = os.path.join(reactions_dir, name)
            rmol = yamol(rinp)
            os.makedirs(subdir, exist_ok=True)
            xyzfiles = create_reaction_xyzs(
                rmol, reactions, work_folder=xtb_temp_dir,
                output_folder=subdir, model_path=model_path,
                n_workers=n_workers, xtb_opt=xtb_opt,
                rcs_filter=rcs_filter, rcs_thresh=rcs_threshold,
                message=False, use_changed_geometry=True,
            )
            print(f"Generated {len(xyzfiles)} reaction xyz files for molecule pair '{name}'.")
            xyzfiles_all.extend(xyzfiles)
    print(f"Total {len(xyzfiles_all)} reaction xyz files generated in '{reactions_dir}'.")



if __name__ == "__main__":
    main()