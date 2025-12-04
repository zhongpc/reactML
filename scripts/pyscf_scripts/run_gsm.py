import os
import argparse

import numpy as np
import yaml
import ase.io
from pyscf import symm
from ase import Atoms
from ase import units
from ase.optimize import FIRE
from ase.units import Hartree, Bohr
from pyGSM.level_of_theories.ase import ASELoT
from pyGSM.potential_energy_surfaces import PES
from pyGSM.utilities.elements import ElementData
from pyGSM.coordinate_systems.topology import Topology
from pyGSM.coordinate_systems.primitive_internals import PrimitiveInternalCoordinates
from pyGSM.coordinate_systems.delocalized_coordinates import DelocalizedInternalCoordinates
from pyGSM.molecule import Molecule
from pyGSM.optimizers import eigenvector_follow, lbfgs
from pyGSM.growing_string_methods import DE_GSM

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
    input_atoms_list = [(ele, coord) for ele, coord in zip(init_atoms.symbols, init_atoms.positions)]
    config["inputfile"] = input_atoms_list  # fake inputfile but input list
    charge = config.get("charge", 0)
    multiplicity = config.get("spin", 0) + 1  # Convert PySCF's 2S (args.spin) to ASE's 2S+1 by adding 1
    if "mlip" in config:
        assert "xc" not in config, "Please do not specify 'xc' when using MLIP."
        for atoms in [init_atoms, final_atoms]:
            atoms.info["charge"] = charge
            atoms.info["spin"] = multiplicity
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

    opt_terminal = config.get("opt_terminal", True)
    fmax = float(config.get("fmax", 0.05))  # in eV/Angstrom
    conv_gmax = fmax * Hartree / Bohr  # convert to Hartree/Bohr
    if opt_terminal:
        init_atoms.calc = calc
        final_atoms.calc = calc
        print("Optimizing the initial structure.")
        with FIRE(init_atoms) as opt0:
            opt0.run(fmax=fmax)
        print("Optimizing the final structure.")
        with FIRE(final_atoms) as opt1:
            opt1.run(fmax=fmax)
        del opt0, opt1

    # set level of theory
    ID = config.get("task_id", 0)
    input_geom = [[ele, *coord] for ele, coord in zip(init_atoms.symbols, init_atoms.positions)]
    lot = ASELoT.from_options(calculator=calc, charge=charge, multiplicity=multiplicity, geom=input_geom, ID=ID)
    # set potential energy surface
    pes = PES.from_options(lot=lot, ad_idx=0)
    # build the topology
    element_data = ElementData()
    elements = [element_data.from_symbol(sym) for sym in init_atoms.symbols]

    topo_react = Topology.build_topology(
        xyz=init_atoms.positions,
        atoms=elements,
    )
    topo_prod = Topology.build_topology(
        xyz=final_atoms.positions,
        atoms=elements,
    )
    # union of bonds
    for bond in topo_prod.edges():
        if bond in topo_react.edges() or bond[::-1] in topo_react.edges():
            continue
        print(f"Adding bond {bond} to the reactant topology.")
        if bond[0] > bond[1]:
            topo_react.add_edge(bond[1], bond[0])
        else:
            topo_react.add_edge(bond[0], bond[1])
    # build primitive internal coordinates
    coordinate_type: str = config.get("coordinate_type", "TRIC")
    coordinate_type = coordinate_type.upper()
    prim_react: PrimitiveInternalCoordinates = PrimitiveInternalCoordinates.from_options(
        xyz=init_atoms.positions,
        atoms=elements,
        topology=topo_react,
        connect=(coordinate_type == "DLC"),
        addtr=(coordinate_type == "TRIC"),
        addcart=(coordinate_type == "HDLC"),
    )
    prim_prod: PrimitiveInternalCoordinates = PrimitiveInternalCoordinates.from_options(
        xyz=final_atoms.positions,
        atoms=elements,
        topology=topo_prod,
        connect=(coordinate_type == "DLC"),
        addtr=(coordinate_type == "TRIC"),
        addcart=(coordinate_type == "HDLC"),
    )
    # add product coords to reactant coords
    prim_react.add_union_primitives(prim_prod)
    
    # delocalized internal coordinates
    deloc_react = DelocalizedInternalCoordinates.from_options(
        xyz=init_atoms.positions,
        atoms=elements,
        connect=(coordinate_type == "DLC"),
        addtr=(coordinate_type == "TRIC"),
        addcart=(coordinate_type == "HDLC"),
        primitives=prim_react,
    )

    # build Molecule objects
    opt_method: str = config.get("opt_method", "LBFGS")
    opt_method = opt_method.lower()
    form_hessian = (opt_method == "eigenvector_follow")
    mol_react = Molecule.from_options(
        geom=input_geom,
        PES=pes,
        coord_obj=deloc_react,
        Form_Hessian=form_hessian,
    )
    num_nodes = config.get("num_nodes", 11)
    mol_prod = Molecule.copy_from_options(
        mol_react,
        xyz=final_atoms.positions,
        new_node_id=num_nodes - 1,
        copy_wavefunction=False,
    )
    # optimizer
    only_climb: bool = config.get("climb", False)
    ediff: float = float(config.get("ediff", 100.))
    emax: float = float(config.get("emax", 1e-6))
    add_node_tol: float = float(config.get("add_node_tol", 0.3))
    max_gsm_steps: int = int(config.get("max_gsm_steps", 500))
    max_opt_steps: int = int(config.get("max_opt_steps", 100))
    print_level: int = int(config.get("print_level", 0))
    step_size: float = float(config.get("step_size", 0.1))
    opt_options = {
        "print_level": print_level,
        "Linesearch": config.get("linesearch", "NoLineSearch"),
        "update_hess_in_bg": not only_climb or opt_method == "lbfgs",
        "conv_Ediff": ediff,
        "conv_gmax": conv_gmax,
        "DMAX": step_size,
        "opt_climb": only_climb,
    }
    if opt_method.lower() == "eigenvector_follow":
        opt = eigenvector_follow.from_options(**opt_options)
    elif opt_method.lower() == "lbfgs":
        opt = lbfgs.from_options(**opt_options)
    else:
        raise NotImplementedError(f"Optimizer {opt_method} not implemented.")
    
    # build GSM
    gsm: DE_GSM = DE_GSM.from_options(
        reactant=mol_react,
        product=mol_prod,
        nnodes=num_nodes,
        CONV_TOL=emax,
        CONV_gmax=conv_gmax,
        CONV_Ediff=ediff,
        ADD_NODE_TOL=add_node_tol,
        growth_direction=0,
        optimizer=opt,
        ID=ID,
        print_level=print_level,
        mp_cores=1,
        interp_method="DLC",
    )
    rtype=1 if only_climb else 0
    gsm.go_gsm(
        max_iters=max_gsm_steps,
        opt_steps=max_opt_steps,
        rtype=rtype,
    )
    atoms_list = []
    for energy, geom in zip(gsm.energies, gsm.geometries):
        atoms = Atoms(
            symbols=init_atoms.symbols,
            positions=[x[1:4] for x in geom],
        )
        atoms.info["energy"] = energy
        atoms_list.append(atoms)
    ase.io.write(f"{filename}_gsm.xyz", atoms_list, format="extxyz")


if __name__ == "__main__":
    main()