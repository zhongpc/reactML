import argparse

import numpy as np
import yaml
import ase.io
from pyscf import symm
from ase import Atoms
from ase import units
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
    if "xc" in config and config["xc"].endswith("3c"):
        xc_3c = config["xc"]
        mf = build_3c_method(config)
    else:
        xc_3c = None
        mf = build_method(config)
    calc = PySCFCalculator(mf=mf, xc_3c=xc_3c)

    # set level of theory
    lot = ASELoT.from_options(calculator=calc, geom=input_atoms_list, ID=0)
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
    coordinate_type = config.get("coordinate_type", "TRIC")
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
    form_hessian = (opt_method.lower() == "eigenvector_follow")
    mol_react = Molecule.from_options(
        geom=input_atoms_list,
        PES=pes,
        coord_obj=deloc_react,
        Form_Hessian=form_hessian,
    )
    num_nodes = config.get("num_nodes", 11)
    mol_prod = Molecule.from_options(
        mol_react,
        xyz=final_atoms.positions,
        new_node_id=num_nodes - 1,
    )
    # optimizer
    only_climb: bool = config.get("only_climb", False)
    conv_gmax: float = config.get("conv_gmax", 0.05)
    conv_energy: float = config.get("conv_energy", 1e-6)
    max_gsm_steps: int = config.get("max_gsm_steps", 500)
    max_opt_steps: int = config.get("max_opt_steps", 100)
    opt_options = {
        "print_level": config.get("print_level", 1),
        "Linesearch": config.get("linesearch", "backtrack"),
        "update_hess_in_bg": not only_climb or opt_method.lower() == "lbfgs",
        "conv_Ediff": conv_energy,
        "conv_gmax": conv_gmax,
        "DMAX": config.get("step_size", 0.1),
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
        CONV_TOL=config.get("gsm_conv_tol", 0.1),
        CONV_gmax=conv_gmax,
        CONV_Ediff=conv_energy,
        ADD_NODE_TOL=config.get("add_node_tol", 0.3),
        growth_direction=0,
        optimizer=opt,
        ID=0,
        print_level=config.get("print_level", 1),
        mp_cores=1,
        inter_method="DLC",
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