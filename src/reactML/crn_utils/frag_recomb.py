"""Create fragments for selected molecules."""
import os
import logging
import json
import itertools
import numpy as np
import copy
from typing import List, Any, Optional, Tuple

from scipy.spatial.distance import cdist
import pymongo
from pymatgen.core.structure import Molecule, Site
from pymatgen.analysis.graphs import MoleculeGraph, ConnectedSite
from pymatgen.analysis.fragmenter import Fragmenter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.local_env import OpenBabelNN
from monty.serialization import loadfn
import ase.io
import ase.data
from ase import Atoms


class FragmentationRecombination:

    MAX_ELECTRONS = 180
    MAX_BONDS = {"C": 4, "P": 5, "S": 6, "O": 2, "N": 3, "B": 5, "Cl": 1, "F": 1}
    AXIS_OF_ROTATION = np.eye(3).tolist()
    delta_distance = 0.05

    def __init__(
        self,
        mol_list: List[Molecule],
        groupname: str = "default_group",
        depth: int = 1,
        bonding_factor_max: float = 1.5,
        bonding_factor_min: float = 0.5,
        n_bonding_factors: int = 10,
        n_angles: int = 100,
        db_collection: Optional[pymongo.collection.Collection] = None,
        **kwargs: Any,
    ):
        """
        Create fragments and recombinations for a list of initial molecule graphs.
        """

        self.mol_list = mol_list
        self.groupname = groupname
        self.depth = depth
        self.bonding_factor_max = bonding_factor_max
        self.bonding_factor_min = bonding_factor_min
        self.n_bonding_factors = n_bonding_factors
        self.n_angles = n_angles
        self.db_collection = db_collection

        self.DEBUG = kwargs.get("debug", False)
        self.output_dir = kwargs.get(
            "output_dir", os.path.join("outputs", "initial_structures")
        )
        os.makedirs(self.output_dir, exist_ok=True)
        self.db_graphs: List[MoleculeGraph] = []
        self.frag_graphs: List[MoleculeGraph] = []
        self.connecting_indices: List[List[int]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.run(*args, **kwargs)

    def run(self, add_monoatomic: bool = False):
        self._get_graphs_from_db()
        self.fragment(add_monoatomic)
        self.recombine()

    def debug(self, *args: Any, **kwargs: Any) -> Any:
        # self._get_graphs_from_db()
        self.fragment()
        # self.recombine()


    def _get_graphs_from_db(self) -> None:
        """Get the graphs already in the database."""
        if self.db_collection is None:
            logging.info("No database collection provided, skipping loading existing graphs")
            return
        self.db_graphs = []
        group_collection = self.db_collection.find({"tags.group": self.groupname})
        for graph in group_collection:
            graph.pop("tags")
            mol_graph = MoleculeGraph.from_dict(graph)
            self.db_graphs.append(mol_graph)
        logging.info(f"Loaded {len(self.db_graphs)} graphs from the database")


    def _exists_in_db(self, molgraph: MoleculeGraph) -> bool:
        """Check if the molecule graph is already in the database."""
        if not self.db_graphs:
            return False
        for db_graph in self.db_graphs:
            # first, check isomorphism
            if not molgraph.isomorphic_to(db_graph):
                continue
            db_molecule: Molecule = db_graph.molecule
            curr_molecule: Molecule = molgraph.molecule
            # second, check charge
            if db_molecule.charge != curr_molecule.charge:
                continue
            # third, check spin multiplicity
            if db_molecule.spin_multiplicity != curr_molecule.spin_multiplicity:
                continue
            # all checks passed, return True
            return True
        # no match found, return False
        return False


    def save_fragments(
        self,
        label: str,
        frag_graphs: List[MoleculeGraph],
        format: str = "xyz",
    ) -> None:
        """Store fragments as xyz files."""
        atoms_list = []
        for frag_graph in frag_graphs:
            atoms: Atoms = AseAtomsAdaptor.get_atoms(frag_graph.molecule)
            atoms.info["charge"] = int(frag_graph.molecule.charge)
            atoms.info["multiplicity"] = int(frag_graph.molecule.spin_multiplicity)
            atoms_list.append(atoms)
        ase.io.write(
            filename=os.path.join(self.output_dir, f"{label}.{format}"),
            images=atoms_list,
            # format=format,
        )

    def fragment(self, add_monoatomic: bool = False) -> None:
        """        
        Generate fragments for all molecules in the list.
        Args:
            add_monoatomic: Whether to add monoatomic species as fragments.
        """
        # create a list of fragmented molecule graphs
        self.frag_graphs = []
        # store the connecting atoms
        self.connecting_indices = []

        for idx, molecule in enumerate(self.mol_list):
            # monoatomic species
            if add_monoatomic and len(molecule) == 1:
                logging.info(f"Adding monoatomic species for molecule {idx}")
                molgraph = MoleculeGraph.with_local_env_strategy(molecule, strategy=OpenBabelNN())
                tags = {
                    "group": self.groupname,
                    "type_structure": "monoatomic",
                    "parent_molecule": molgraph.molecule.as_dict(),
                    "idx_connecting_atoms": 0,
                    "idx_mol": 0,
                }
                label = "fragments_" + "_".join(molecule.formula.split())

                self.frag_graphs.append(molgraph)
                self._append_graph_to_db(molgraph, tags)
                self.connecting_indices.append([0])
            # fragment larger molecules
            else:
                logging.info(f"Fragmenting molecule {idx} with formula {molecule.formula} up to depth {self.depth}")
                fragmenter = Fragmenter(molecule, depth=self.depth, open_rings=True)
                logging.info(f"Number of fragments: {fragmenter.total_unique_fragments}")

                for key, frag_graphs in fragmenter.unique_frag_dict.items():
                    label = "fragments_" + "_".join(key.split())
                    self.save_fragments(label, frag_graphs, format="xyz")
                    self.frag_graphs.extend(frag_graphs)

                    # Determine the connecting atoms in the molecule
                    for idx_mol, frag_graph in enumerate(frag_graphs):
                        frag_graph: MoleculeGraph
                        connected_sites = self._generate_connected_sites(frag_graph)
                        tags = {
                            "group": self.groupname,
                            "type_structure": "fragment",
                            "parent_molecule": frag_graph.molecule.as_dict(),
                            "idx_connecting_atoms": connected_sites,
                            "idx_mol": idx_mol,
                        }
                        self._append_graph_to_db(frag_graph, tags)
        if self.DEBUG:
            logging.debug(f"Number of fragment graphs: {len(self.frag_graphs)}")
            logging.debug(f"Number of connecting indices: {len(self.connecting_indices)}")
            logging.debug(f"Connecting indices: {self.connecting_indices}")


    def _generate_connected_sites(self, frag_graph: MoleculeGraph) -> List[int]:
        """For all fragments, generate a list of connecting atoms."""
        molecule: Molecule = frag_graph.molecule
        connected_sites = []
        for idx in range(len(molecule)):
            if not self._is_valid_species(molecule, idx):
                continue
            if not self._has_free_bonds(
                frag_graph.get_connected_sites(idx), str(molecule[idx].specie)
            ):
                continue
            connected_sites.append(idx)
        self.connecting_indices.append(connected_sites)
        return connected_sites


    def _append_graph_to_db(self, molgraph: MoleculeGraph, tags: dict) -> None:
        """Put the structure in the initial structure database."""
        if self.DEBUG:
            logging.debug("Debug mode: not adding to database")
            return
        if self.db_collection is None:
            logging.info("No database collection provided, skipping adding to database")
            return
        if self._exists_in_db(molgraph):
            logging.warning("Structure already present in database")
            return
        molgraph_dict = molgraph.as_dict()
        serialize_molecule_graph(molgraph_dict)
        molgraph_dict["tags"] = tags
        self.db_collection.insert_one(molgraph_dict)


    def _has_valid_tot_charge(self, frag_graph1: MoleculeGraph, frag_graph2: MoleculeGraph) -> bool:
        """Check if the charge passes the criteria."""
        molecule1: Molecule = frag_graph1.molecule
        molecule2: Molecule = frag_graph2.molecule
        tot_charge = molecule1.charge + molecule2.charge
        if tot_charge not in [-2, -1, 0, 1, 2]:
            logging.debug(
                f"Total charge of {tot_charge} with {molecule1.charge} and {molecule2.charge} is not between -2 and 2, rejected"
            )
            return False
        logging.debug(
            f"Total charge of {tot_charge} with {molecule1.charge} and {molecule2.charge} is accepted"
        )
        return True

    def _has_valid_tot_electrons(self, frag_graph1: MoleculeGraph, frag_graph2: MoleculeGraph) -> bool:
        """Check if the number of electrons exceeds our computational capacity."""
        molecule1: Molecule = frag_graph1.molecule
        molecule2: Molecule = frag_graph2.molecule
        tot_electrons = molecule1.nelectrons + molecule2.nelectrons
        if tot_electrons < self.MAX_ELECTRONS:
            logging.debug(f"Electrons {tot_electrons} is accepted")
            return True
        logging.debug(f"Electrons {tot_electrons} exceeds maximum of {self.MAX_ELECTRONS}, rejected")
        return False

    def _is_valid_species(self, molecule: Molecule, index: int) -> bool:
        """Check if the species is suitable to be added"""
        if str(molecule[index].specie) in self.MAX_BONDS:
            return True
        # Also pass isolated instances of H, Li, Mg, Ca
        elif str(molecule[index].specie) in ["H", "Li", "Mg", "Ca"] and len(molecule) == 1:
            return True
        return False

    def _has_free_bonds(self, connected_sites: List[ConnectedSite], element: str) -> bool:
        """Check if the atom has enough free bonds."""
        # (not necessarily correct) these will generate non-bonded connections, like hydrogen bonds
        if element in ["H", "Li", "Mg", "Ca"]:
            return True

        # check if the atom has enough free bonds
        tot_connected_sites = len(connected_sites)

        # remove metal sites from the tot_connected_sites
        for connected_site in connected_sites:
            site: Site = connected_site.site
            if str(site.specie) in ["Li", "Mg", "Ca"]:
                tot_connected_sites -= 1

        if tot_connected_sites < self.MAX_BONDS[element]:
            return True
        
        return False
        

    def recombine(self):
        """
        Create recombinants of the molecule.
        """
        num_fragments = len(self.frag_graphs)
        idx_fragments = list(range(num_fragments))

        # generate the combinations of fragments
        combinations = itertools.combinations(idx_fragments, 2)

        # iterate over the combinations
        n_tot, n_accepted = 0, 0
        for idx, (i_comb1, i_comb2) in enumerate(combinations):

            # Get the two fragments
            frag_graph1 = self.frag_graphs[i_comb1]
            frag_graph2 = self.frag_graphs[i_comb2]
            # Check charge
            if not self._has_valid_tot_charge(frag_graph1, frag_graph2):
                continue
            if not self._has_valid_tot_electrons(frag_graph1, frag_graph2):
                continue

            # iterate over the connecting atoms and connect the two graphs
            # at the appropriate connecting atom
            heavy_atoms_sites1 = self.connecting_indices[i_comb1]
            heavy_atoms_sites2 = self.connecting_indices[i_comb2]

            # iterate over the potential connecting atoms
            for site1 in heavy_atoms_sites1:
                for site2 in heavy_atoms_sites2:
                    # add to the total number of recombinations
                    n_tot += 1
                    # create a new graph with both fragments
                    combined_mol_graph = self.combine_frag_graphs(
                        frag_graph1, frag_graph2, idx, site1, site2
                    )
                    if combined_mol_graph is not None:

                        label = f"combination_{idx}_s{site1}_s{site2 + len(frag_graph1.molecule)}"
                        self.save_fragments(label, [combined_mol_graph])

                        # Generate the tags to store in mongodb
                        frag1_dict = frag_graph1.as_dict()
                        frag2_dict = frag_graph2.as_dict()

                        serialize_molecule_graph(frag1_dict)
                        serialize_molecule_graph(frag2_dict)
                        tags = {
                            "group": self.groupname,
                            "type_structure": "recombination",
                            "frag1_graphs": frag1_dict,
                            # "connecting_atoms_frag1": heavy_atoms_sites1,
                            "idx1_connecting_atoms": site1,
                            "frag2_graphs": frag2_dict,
                            # "connecting_atoms_frag2": heavy_atoms_sites2,
                            "idx2_connecting_atoms": site2,
                        }
                        self._append_graph_to_db(combined_mol_graph, tags)

                        # The total number of recombinations
                        n_accepted += 1

        logging.info(f"Total number of recombinations: {n_tot}")
        logging.info(
            f"Accepted number of recombinations: {n_accepted}"
        )

    def vdw_radii(self, symbol: str) -> float:
        """Repurpose ASEs van der Waals radii."""
        atomic_number = ase.data.atomic_numbers[symbol]
        return ase.data.vdw_radii[atomic_number]

    def covalent_radii(self, symbol: str) -> float:
        """Repurpose ASEs covalent radii."""
        atomic_number = ase.data.atomic_numbers[symbol]
        return ase.data.covalent_radii[atomic_number]

    def _is_overlapped(
        self,
        molecule1: Molecule,
        molecule2: Molecule,
        bonding_factor: float,
        check_min: bool = False,
        connected_sites: Optional[List[int]] = None,
    ):
        """Check if the two molecules overlap."""
        # Get the positions of the atoms
        positions1 = molecule1.cart_coords
        positions2 = molecule2.cart_coords

        species1 = molecule1.species
        species2 = molecule2.species
        species1 = [a.symbol for a in species1]
        species2 = [a.symbol for a in species2]

        # generate the sum over covalent radii matrix
        dist_matrix = cdist(positions1, positions2)
        covalent_radii1 = np.array([self.covalent_radii(s) for s in species1])
        covalent_radii2 = np.array([self.covalent_radii(s) for s in species2])
        radii_matrix = covalent_radii1[:, np.newaxis] + covalent_radii2[np.newaxis, :]
        # this is the bond between the two atoms, divide by bonding factor
        if connected_sites is not None:
            i, j = connected_sites
            radii_matrix[i, j] /= bonding_factor 

        # if the distance is smaller than the sum of the covalent radii, there is an overlap
        if np.any(dist_matrix < radii_matrix):
            return True

        # check if the connected sites are one of the minimum distances
        if check_min and connected_sites is not None:
            i, j = connected_sites
            intended_bond_dist = dist_matrix[i, j]
            sorted_dist = np.sort(dist_matrix.flatten(), axis=None)
            # must within the 10 shortest distances
            k_shortest = 10 if len(sorted_dist) > 10 else len(sorted_dist)
            if intended_bond_dist <= sorted_dist[k_shortest - 1]:
                # The intended bond is among the closest atoms
                logging.debug(
                    f"Current min distance: {intended_bond_dist} is among the closest distances."
                )
                return False
            else:
                # The intended bond is not among the closest atoms
                logging.debug(
                    f"Current min distance: {intended_bond_dist} is NOT among the closest distances."
                )
                return True
        return False
    

    def rotate_molecules(
        self,
        frag_graph1: MoleculeGraph,
        frag_graph2: MoleculeGraph,
        axis: List[float],
        angle: float,
        bonding_factor: float,
        intermediate_molecules: List[Molecule],
        site1: int,
        site2: int,
        rotate_first: bool = True,
    ) -> Tuple[bool, Optional[MoleculeGraph], Optional[MoleculeGraph]]:

        frag_copy1 = copy.deepcopy(frag_graph1)
        mol_copy1: Molecule = frag_copy1.molecule
        frag_copy2 = copy.deepcopy(frag_graph2)
        mol_copy2: Molecule = frag_copy2.molecule

        # Rotate the first molecule first
        if rotate_first:
            rotation_indices = list(range(len(mol_copy1)))
            mol_copy1.rotate_sites(
                indices=rotation_indices,
                theta=angle,
                axis=axis,
                anchor=[0, 0, 0],
            )
        else:
            rotation_indices = list(range(len(mol_copy2)))
            mol_copy2.rotate_sites(
                indices=rotation_indices,
                theta=angle,
                axis=axis,
                anchor=[0, 0, 0],
            )

        # store the generated molecule based on copy_2
        inter_graph = copy.deepcopy(frag_copy1)
        for site in mol_copy2:
            inter_graph.molecule.append(site.specie, site.coords)
        intermediate_molecules.append(inter_graph)

        if not self._is_overlapped(
            mol_copy1,
            mol_copy2,
            bonding_factor,
            check_min=True,
            connected_sites=[site1, site2],
        ):
            return True, frag_copy1, frag_copy2
        return False, None, None


    def combine_frag_graphs(
        self,
        frag_graph1: MoleculeGraph,
        frag_graph2: MoleculeGraph,
        idx: int,
        site1: int,
        site2: int,
    ) -> MoleculeGraph:
        """Combine molecular graphs and generate the starting structures for a DFT calculation."""

        # commonly used information in this method
        molecule1: Molecule = frag_graph1.molecule
        molecule2: Molecule = frag_graph2.molecule
        cspec_1 = str(molecule1[site1].specie)
        cspec_2 = str(molecule2[site2].specie)
        sum_radii = self.covalent_radii(cspec_1) + self.covalent_radii(cspec_2)

        # start by moving the first molecule to the origin based on the coordinates
        # of the connecting atom in the molecule.
        frag_copy1 = copy.deepcopy(frag_graph1)
        mol_copy1: Molecule = frag_copy1.molecule
        mol_copy1.translate_sites(
            list(range(len(frag_copy1.molecule))),
            vector=-mol_copy1[site1].coords,
        )

        # get the index of the connecting atoms
        label = f"intermediate_combination_{idx}_s{site1}_s{site2 + len(mol_copy1)}"

        # store the intermediate molecules during the rotation
        intermediate_molecules = []
        found_structure = False

        # iterate over the placement of the atoms and the angle of rotation to find feasible structures.
        range_bonding = np.linspace(
            self.bonding_factor_max, self.bonding_factor_min, self.n_bonding_factors
        )
        logging.debug(f"Range bonding: {range_bonding}")

        for bonding_factor in range_bonding:

            # generate the second molecule at the correct distance and angle
            # unit vector from center of molecule 1 to its connecting site
            vec_center_to_site1 = mol_copy1.center_of_mass - mol_copy1[site1].coords
            # in case the two atoms are at the same position, set an arbitrary vector
            length_vec = np.linalg.norm(vec_center_to_site1)
            if length_vec < 1e-6:
                vec_unit = np.array([1.0, 0.0, 0.0])
            else:
                vec_unit = vec_center_to_site1 / np.linalg.norm(vec_center_to_site1)
            translate_vec = (
                -np.array(molecule2[site2].coords)
                + (sum_radii * bonding_factor + self.delta_distance) * vec_unit
            )
            frag_copy2 = copy.deepcopy(frag_graph2)
            mol_copy2: Molecule = frag_copy2.molecule
            mol_copy2.translate_sites(
                list(range(len(mol_copy2))),
                vector=translate_vec,
            )

            # Now iterate over the angles of rotation to find the correct angle
            # which does not overlap with the first molecule
            for axis in self.AXIS_OF_ROTATION:

                logging.debug(f"Rotate along axis: {axis}")
                # Iterate over angles.
                for angle in np.linspace(0, 2 * np.pi, self.n_angles):

                    angle_in_degrees = np.degrees(angle)
                    logging.debug(f"Rotating molecule to {angle_in_degrees} degrees")

                    # Rotate the first molecule first
                    found_structure, frag_rotated1, frag_rotated2 = self.rotate_molecules(
                        frag_graph1=frag_copy1,
                        frag_graph2=frag_copy2,
                        axis=axis,
                        angle=angle,
                        bonding_factor=bonding_factor,
                        intermediate_molecules=intermediate_molecules,
                        site1=site1,
                        site2=site2,
                        rotate_first=True,
                    )
                    if found_structure:
                        frag_copy1, frag_copy2 = frag_rotated1, frag_rotated2
                        mol_copy1, mol_copy2 = frag_copy1.molecule, frag_copy2.molecule
                        break
                    else:
                        logging.debug(
                            f"No rotation found for first molecule rotation that does not overlap for axis {axis}"
                        )
                    # Rotate the second molecule first
                    found_structure, frag_rotated1, frag_rotated2 = self.rotate_molecules(
                        frag_graph1=frag_copy1,
                        frag_graph2=frag_copy2,
                        axis=axis,
                        angle=angle,
                        bonding_factor=bonding_factor,
                        intermediate_molecules=intermediate_molecules,
                        site1=site1,
                        site2=site2,
                        rotate_first=False,
                    )
                    if found_structure:
                        frag_copy1, frag_copy2 = frag_rotated1, frag_rotated2
                        mol_copy1, mol_copy2 = frag_copy1.molecule, frag_copy2.molecule
                        break
                    else:
                        logging.debug(
                            f"No rotation found for second molecule rotation that does not overlap for axis {axis}"
                        )
                if found_structure:
                    break
            if found_structure:
                break
            else:
                logging.debug(
                    f"No displacement by {bonding_factor} leads to no overlap."
                )
        else:
            # No operation worked, write out the intermediate molecules
            label = "failed_" + label
            self.save_fragments(label, intermediate_molecules, format="xyz")
            logging.warning(
                f"Could not find a rotation angle that minimizes overlap for {label}"
            )
            return

        if len(intermediate_molecules) > 1:
            self.save_fragments(label, intermediate_molecules, format="xyz")

        # Finally, combine the two molecules
        for site in mol_copy2:
            frag_copy1.insert_node(len(mol_copy1), site.specie, site.coords)
        for edge in frag_copy2.graph.edges():
            side_1 = edge[0] + len(molecule1)
            side_2 = edge[1] + len(molecule1)
            frag_copy1.add_edge(side_1, side_2)
        mol_copy1.set_charge_and_spin(
            molecule1.charge + molecule2.charge
        )
        frag_copy1.add_edge(site1, site2 + len(molecule1))

        return frag_copy1


def get_molecule_list_from_LIBE(
    libe_ids, db, output_dir: str = "outputs"
) -> List[Molecule]:
    """Generate a list of Molecules based on the LIBE IDs."""

    molecule_list = []
    results_collection = db.mp_summary

    for libe_id in libe_ids:
        # Find the molecule entry based on the libe_id
        clean_libe_id = libe_id.replace("libe-", "")
        result = results_collection.find_one({"molecule_id": int(clean_libe_id)})
        molecule = Molecule.from_dict(result["molecule"])
        molecule_list.append(molecule)
        # Write the molecule to a file
        molecule.to(filename=os.path.join(output_dir, f"{clean_libe_id}.xyz"))

    return molecule_list


def get_molecule_from_json(libe_ids: List[str], json_filename) -> List[Molecule]:
    """Get the molecule list from the json file."""

    molecule_list = []

    # Read json file
    with open(json_filename, "r") as f:
        data_dict = json.load(f)

    for data in data_dict:
        if data["molecule_id"] in libe_ids:
            molecule = Molecule.from_dict(data["molecule"])
            molecule_list.append(molecule)
            # Write the molecule to a file
            molecule.to(filename=os.path.join("outputs", f"{data['molecule_id']}.xyz"))
            logging.debug(f"Added molecule {molecule} to the to-fragment list")

    return molecule_list

def get_all_molecule_from_json(json_filename) -> List[Molecule]:
    """Get the molecule list from the json file."""

    molecule_list = []

    # Read json file
    with open(json_filename, "r") as f:
        data_dict = json.load(f)

    for data in data_dict:
        molecule = Molecule.from_dict(data["molecule"])
        molecule_list.append(molecule)
        # Write the molecule to a file
        molecule.to(filename=os.path.join("outputs", f"{data['molecule_id']}.xyz"))
        logging.debug(f"Added molecule {molecule} to the to-fragment list")

    return molecule_list


def get_molecule_graph_from_json(libe_ids: List[str], json_filename) -> List[Molecule]:
    """Get the molecule graph list from the json file."""

    molecule_list = []

    # Read json file
    data_dict = loadfn(json_filename)

    for data in data_dict:
        if data["molecule_id"] in libe_ids:
            molecule_graph = data["molecule_graph"]
            molecule_list.append(molecule_graph)
            # Write the molecule to a file
            logging.debug(f"Added molecule {molecule_graph} to the to-fragment list")

    return molecule_list


def serialize_molecule_graph(molgraph_dict: dict) -> None:
    """Convert numpy arrays to lists in the molecule graph dict for serialization."""
    graph_dict: dict = molgraph_dict["graphs"]
    nodes = graph_dict.pop("nodes")
    # Convert arrays to lists
    for node in nodes:
        array_xyz: np.ndarray = node["coords"]
        node["coords"] = array_xyz.tolist()
    molgraph_dict["graphs"]["nodes"] = nodes

