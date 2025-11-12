"""Create fragments for selected molecules."""
import os
import logging
import itertools
import numpy as np
import copy
from typing import Any, Optional, Union, List, Tuple, Dict

from scipy.spatial.distance import cdist
from pymatgen.core.structure import Molecule, Site
from pymatgen.analysis.graphs import MoleculeGraph, ConnectedSite
from pymatgen.analysis.fragmenter import Fragmenter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.babel import BabelMolAdaptor
from pymatgen.analysis.local_env import OpenBabelNN
import ase.io
import ase.data
from ase import Atoms


class FragmentationRecombination:

    MAX_ELECTRONS = 180
    MAX_BONDS = {"C": 4, "P": 5, "S": 6, "O": 2, "N": 3, "B": 5, "Cl": 1, "F": 1}
    AXIS_OF_ROTATION = np.eye(3).tolist()
    def __init__(
        self,
        molecules: List[Molecule],
        frag_depths: Union[int, List[int]] = 1,
        bonding_scale_max: float = 1.0,
        bonding_scale_min: float = 1.0,
        n_bonding_scales: int = 1,
        n_angles: int = 100,
        early_stop_angular: bool = True,  # whether to stop at first found angle
        extra_max_bonds: Optional[Dict[str, int]] = None,
        localopt: bool = False,
        output_dir: str = "./struct",
        **kwargs: Any,
    ):
        """
        Create fragments and recombinations for a list of initial molecule graphs.
        """

        # read the input arguments
        self.molecules = molecules
        if isinstance(frag_depths, int):
            self.frag_depths = [frag_depths for _ in range(len(molecules))]
        elif isinstance(frag_depths, list):
            assert len(frag_depths) == len(molecules), "Length of frag_depths must match number of molecules"
            self.frag_depths = frag_depths
        else:
            raise ValueError("frag_depths must be an int or a list of ints")
        self.bonding_scale_max = bonding_scale_max
        self.bonding_scale_min = bonding_scale_min
        self.n_bonding_scales = n_bonding_scales
        self.n_angles = n_angles
        self.early_stop_angular = early_stop_angular
        if extra_max_bonds is not None:
            self.MAX_BONDS.update(extra_max_bonds)
        self.localopt = localopt
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.DEBUG = kwargs.get("debug", False)

        # initialize fragments list
        self.frag_graphs: List[MoleculeGraph] = []
        self.connecting_indices: List[List[int]] = []

    def run(self, add_monoatomic: bool = False):
        self.fragment(add_monoatomic)
        self.recombine()

    def debug(self, *args: Any, **kwargs: Any) -> Any:
        self.fragment()
        # self.recombine()

    def save_fragments(
        self,
        label: str,
        frag_graphs: List[MoleculeGraph],
        localopt: bool = False,
        format: str = "xyz",
    ) -> None:
        """Store fragments as xyz files."""
        atoms_list = []
        for frag_graph in frag_graphs:
            if localopt:
                babel_mol = BabelMolAdaptor.from_molecule_graph(frag_graph)
                babel_mol.localopt(forcefield="UFF")
                molecule = babel_mol.pymatgen_mol
            else:
                molecule: Molecule = frag_graph.molecule
            atoms: Atoms = AseAtomsAdaptor.get_atoms(molecule)
            atoms.info["charge"] = int(molecule.charge)
            atoms.info["multiplicity"] = int(molecule.spin_multiplicity)
            atoms_list.append(atoms)
        ase.io.write(
            filename=os.path.join(self.output_dir, f"{label}.{format}"),
            images=atoms_list,
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

        for i_mol, (molecule, frag_depth) in enumerate(zip(self.molecules, self.frag_depths)):
            # monoatomic species
            if add_monoatomic and len(molecule) == 1:
                logging.info(f"Adding monoatomic species for molecule {i_mol}")
                molgraph = MoleculeGraph.with_local_env_strategy(molecule, strategy=OpenBabelNN())
                label = "fragments_" + "_".join(molecule.formula.split())
                self.frag_graphs.append(molgraph)
                self.connecting_indices.append([0])
            # fragment larger molecules
            else:
                logging.info(f"Fragmenting molecule {i_mol} with formula {molecule.formula} up to depth {frag_depth}")
                fragmenter = Fragmenter(molecule, depth=frag_depth, open_rings=True)
                logging.info(f"Number of fragments: {fragmenter.total_unique_fragments}")
                for key, frag_graphs in fragmenter.unique_frag_dict.items():
                    label = "fragments_" + "_".join(key.split())
                    self.save_fragments(label, frag_graphs, format="xyz")
                    self.frag_graphs.extend(frag_graphs)
                    # determine the connecting atoms in the molecule
                    for frag_graph in frag_graphs:
                        connected_sites = self._generate_connected_sites(frag_graph)
                        self.connecting_indices.append(connected_sites)
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
        return connected_sites

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
        # generate the combinations of fragments
        frag_indices = list(range(len(self.frag_graphs)))
        frag_pairs = itertools.combinations(frag_indices, 2)

        # iterate over the combinations
        n_tot, n_accepted = 0, 0
        for i_comb, (i_frag1, i_frag2) in enumerate(frag_pairs):
            # get the two fragments
            frag_graph1 = self.frag_graphs[i_frag1]
            frag_graph2 = self.frag_graphs[i_frag2]
            # check charge and electron number criteria
            if not self._has_valid_tot_charge(frag_graph1, frag_graph2):
                continue
            if not self._has_valid_tot_electrons(frag_graph1, frag_graph2):
                continue

            # iterate over the connecting atoms and connect the two graphs at the appropriate connecting atom
            frag1_sites = self.connecting_indices[i_frag1]
            frag2_sites = self.connecting_indices[i_frag2]
            # iterate over the potential connecting atoms
            for site1 in frag1_sites:
                for site2 in frag2_sites:
                    # add to the total number of recombinations
                    n_tot += 1
                    # create a new graph with both fragments
                    combined_mol_graph = self.combine_frag_graphs(
                        frag_graph1, frag_graph2, i_comb, site1, site2
                    )
                    if combined_mol_graph is not None:
                        label = f"combination_{i_comb}_s{site1}_s{site2 + len(frag_graph1.molecule)}"
                        self.save_fragments(label, [combined_mol_graph], localopt=self.localopt)
                        # The total number of recombinations
                        n_accepted += 1

        logging.info(f"Total number of recombinations: {n_tot}")
        logging.info(f"Accepted number of recombinations: {n_accepted}")

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
        connected_sites: Optional[List[int]] = None,
        radius_type: str = "covalent",
        return_avg_dist: bool = False,
    ) -> Union[bool, Tuple[bool, float]]:
        """Check if the two molecules overlap."""
        # Get the positions of the atoms
        is_overlapped = False

        positions1 = molecule1.cart_coords
        positions2 = molecule2.cart_coords

        species1 = molecule1.species
        species2 = molecule2.species
        species1 = [a.symbol for a in species1]
        species2 = [a.symbol for a in species2]

        # generate the sum over covalent radii matrix
        dist_matrix = cdist(positions1, positions2)
        if radius_type == "vdw":
            radii1 = np.array([self.vdw_radii(s) for s in species1])
            radii2 = np.array([self.vdw_radii(s) for s in species2])
        elif radius_type == "covalent":
            radii1 = np.array([self.covalent_radii(s) for s in species1])
            radii2 = np.array([self.covalent_radii(s) for s in species2])
        else:
            raise ValueError(f"Radius type {radius_type} not recognized")
        radii_matrix = radii1[:, np.newaxis] + radii2[np.newaxis, :]
        # distance between connect sites is not included in the overlap check
        if connected_sites is not None:
            i, j = connected_sites
            radii_matrix[i, j] = 0.0  # means any distance is acceptable

        # if the distance is smaller than the sum of the covalent radii, there is an overlap
        if np.any(dist_matrix < radii_matrix):
            is_overlapped = True
        
        if return_avg_dist:
            avg_dist = np.mean(dist_matrix).item()
            return is_overlapped, avg_dist
        return is_overlapped
    

    def rotate_molecules(
        self,
        frag_graph1: MoleculeGraph,
        frag_graph2: MoleculeGraph,
        axis: List[float],
        angle: float,
        intermediate_molecules: List[Molecule],
        site1: int,
        site2: int,
        rotate_first: bool = True,
    ) -> Tuple[bool, float, Optional[MoleculeGraph], Optional[MoleculeGraph]]:

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

        is_overlapped, avg_dist = self._is_overlapped(
            mol_copy1,
            mol_copy2,
            connected_sites=[site1, site2],
            radius_type="covalent",
            return_avg_dist=True,
        )
        found_structure = not is_overlapped  # if not overlapped, we found a structure
        return found_structure, avg_dist, frag_copy1, frag_copy2


    def combine_frag_graphs(
        self,
        frag_graph1: MoleculeGraph,
        frag_graph2: MoleculeGraph,
        index: int,
        site1: int,
        site2: int,
        early_stop_angular: bool = True,
        save_intermediate: bool = False,
    ) -> MoleculeGraph:
        """Combine molecular graphs and generate the starting structures for a DFT calculation."""

        # commonly used information in this method
        molecule1: Molecule = frag_graph1.molecule
        molecule2: Molecule = frag_graph2.molecule
        cspec1 = str(molecule1[site1].specie)
        cspec2 = str(molecule2[site2].specie)
        sum_radii = self.covalent_radii(cspec1) + self.covalent_radii(cspec2)

        # start by moving the first molecule to the origin based on the coordinates
        # of the connecting atom in the molecule.
        frag_copy1 = copy.deepcopy(frag_graph1)
        mol_copy1: Molecule = frag_copy1.molecule
        mol_copy1.translate_sites(
            list(range(len(frag_copy1.molecule))),
            vector=-mol_copy1[site1].coords,
        )

        # get the index of the connecting atoms
        label = f"intermediate_{index}_s{site1}_s{site2 + len(mol_copy1)}"

        # store the intermediate molecules during the rotation
        intermediate_molecules = []
        found_structure = False

        # iterate over the placement of the atoms and the angle of rotation to find feasible structures.
        bonding_scales = np.linspace(
            self.bonding_scale_min, self.bonding_scale_max, self.n_bonding_scales
        )
        logging.debug(f"Range bonding: {bonding_scales}")

        for bonding_scale in bonding_scales:

            # record the structure with minimal overlap
            avg_dist = 0.0

            # generate the second molecule at the correct distance and angle
            # unit vector from center of molecule 1 to its connecting site
            vec_center_to_site1 = mol_copy1[site1].coords - mol_copy1.center_of_mass
            # in case the two atoms are at the same position, set an arbitrary vector
            length_vec = np.linalg.norm(vec_center_to_site1)
            if length_vec < 1e-6:
                vec_unit = np.array([1.0, 0.0, 0.0])
            else:
                vec_unit = vec_center_to_site1 / np.linalg.norm(vec_center_to_site1)
            translate_vec = (
                -np.array(molecule2[site2].coords)
                + (sum_radii * bonding_scale) * vec_unit
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
                for angle in np.linspace(0, 2*np.pi, self.n_angles):
                    angle_in_degrees = np.degrees(angle)
                    logging.debug(f"Rotating molecule to {angle_in_degrees} degrees")
                    # Rotate the first molecule first
                    found_structure, curr_avg_dist, frag_rotated1, frag_rotated2 = self.rotate_molecules(
                        frag_graph1=frag_copy1,
                        frag_graph2=frag_copy2,
                        axis=axis,
                        angle=angle,
                        intermediate_molecules=intermediate_molecules,
                        site1=site1,
                        site2=site2,
                        rotate_first=True,
                    )
                    if found_structure and curr_avg_dist > avg_dist:
                        frag_copy1, frag_copy2 = frag_rotated1, frag_rotated2
                        mol_copy1, mol_copy2 = frag_copy1.molecule, frag_copy2.molecule
                        avg_dist = curr_avg_dist
                    # Rotate the second molecule first
                    found_structure, curr_avg_dist, frag_rotated1, frag_rotated2 = self.rotate_molecules(
                        frag_graph1=frag_copy1,
                        frag_graph2=frag_copy2,
                        axis=axis,
                        angle=angle,
                        intermediate_molecules=intermediate_molecules,
                        site1=site1,
                        site2=site2,
                        rotate_first=False,
                    )
                    if found_structure and curr_avg_dist > avg_dist:
                        frag_copy1, frag_copy2 = frag_rotated1, frag_rotated2
                        mol_copy1, mol_copy2 = frag_copy1.molecule, frag_copy2.molecule
                        avg_dist = curr_avg_dist
                # stop rotation searching early
                if found_structure and early_stop_angular:
                    break
            # stop bonding scale searching early
            if found_structure:
                break
        if not found_structure:
            logging.info(f"Could not find a rotation angle that minimizes overlap for {label}")
            if save_intermediate:
                logging.info(f"Saving intermediate structures for failed {label}")
                label = "failed_" + label
                self.save_fragments(label, intermediate_molecules, format="xyz")
            return

        if len(intermediate_molecules) > 1 and save_intermediate:
            logging.info(f"Saving intermediate structures for {label}")
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
