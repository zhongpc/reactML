from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN

from mrnet.core.mol_entry import MoleculeEntry, MoleculeEntryError
from mrnet.core.reactions import bucket_mol_entries

from typing import Dict, List, Tuple
import random
import pymongo
import json



def mkdir(path: str):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print("Folder exists")
    return path


def read_db_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # If not found, return None
    return data


def instance_mongodb(config: Dict) -> pymongo.database.Database:
    """Create an instance of mongodb from reading the yaml config."""
    # set default login info to None
    host = config.get("host", None)
    username = config.get("username", None)
    password = config.get("password", None)
    authSource = config.get("authSource", "admin")
    client = pymongo.MongoClient(
        host=host,
        username=username, 
        password=password, 
        authSource=authSource, 
    )
    database_name = config["database"]
    db = client[database_name]
    return db


def get_all_graphs(data):
    """Extract all graphs from the data."""
    all_graphs = []
    all_ids = []
    for entry in data:
        all_graphs.append(entry["molecule_graph"])
        all_ids.append(entry["molecule_id"])
    return all_graphs, all_ids


def get_all_graphs_from_collection(collection):
    """Extract all graphs from the data."""
    all_graphs = []
    for entry in collection:
        molecule = entry["input"]["initial_molecule"]
        molecule = Molecule.from_dict(molecule)
        # Create the molecule graph
        molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())
        all_graphs.append(molecule_graph)
    return all_graphs


def check_already_completed(molecule_graph, all_graphs):
    """Check if the molecule graph is already completed."""
    for graph in all_graphs:
        if molecule_graph.isomorphic_to(graph):
            if molecule_graph.molecule.charge == graph.molecule.charge:
                if (
                    molecule_graph.molecule.spin_multiplicity
                    == graph.molecule.spin_multiplicity
                ):
                    return True
    return False



def get_required_schema(
    molecule_graph: MoleculeGraph,
    recombination_data_list: List[Dict],
    molecule_id: str,
    temperature: float = 298.15,
) -> Dict:
    """Convert molecule graph to required schema."""

    # Get the recombination data
    recombination_data, recombination_data_quantities = recombination_data_list

    data = {}
    data["bonds"] = {}
    data["charge"] = molecule_graph.molecule.charge
    data["chemical_system"] = recombination_data["chemsys"]
    data["formula_alphabetical"] = recombination_data["formula_alphabetical"]
    data["molecule_graph"] = molecule_graph.as_dict()
    data["molecule"] = molecule_graph.molecule.as_dict()
    data["molecule_id"] = molecule_id
    data["number_atoms"] = len(molecule_graph.molecule)
    data["number_elements"] = len(molecule_graph.molecule.composition.elements)
    data["partial_charges"] = {}
    data["partial_charges"]["resp"] = recombination_data["output"]["resp"]
    data["partial_charges"]["mulliken"] = recombination_data["output"]["mulliken"]
    data["partial_charges"]["nbo"] = recombination_data_quantities["output"]["nbo"]
    data["partial_spins"] = {}
    data["partial_spins"]["resp"] = None
    data["partial_spins"]["mulliken"] = None
    data["partial_spins"]["nbo"] = None
    data["point_group"] = recombination_data["pointgroup"]
    data["redox"] = {}
    data["redox"]["electron_affinity_eV"] = None
    data["redox"]["ionization_energy_eV"] = None
    data["species"] = recombination_data["calcs_reversed"][0]["species"]
    data["spin_multiplicity"] = molecule_graph.molecule.spin_multiplicity
    data["thermo"] = {}

    # electronic_energy = recombination_data["output"]["final_energy"]
    # electronic_energy_eV = electronic_energy * ase_units.Hartree
    # total_enthalpy = recombination_data["calcs_reversed"][0]["total_enthalpy"]
    # total_enthalpy_eV = total_enthalpy * kcal_per_mol_to_eV
    # zpe = recombination_data["calcs_reversed"][0]["ZPE"]
    # zpe_eV = zpe * kcal_per_mol_to_eV
    # total_entropy = recombination_data["calcs_reversed"][0]["total_entropy"]
    # total_entropy_eV = total_entropy * kcal_per_mol_to_eV / 1000
    # free_energy_eV = (
    #     electronic_energy_eV + total_enthalpy_eV - total_entropy_eV * temperature
    # )

    # data["thermo"]["eV"] = {
    #     "electronic_energy": electronic_energy_eV,
    #     "free_energy": free_energy_eV,
    #     "total_enthalpy": total_enthalpy_eV,
    #     "total_entropy": total_entropy_eV,
    # }
    # data["thermo"]["raw"] = {
    #     "electronic_energy_Ha": electronic_energy,
    #     "rotational_enthalpy_kcal/mol": recombination_data["calcs_reversed"][0][
    #         "rot_enthalpy"
    #     ],
    #     "rotational_entropy_cal/molK": recombination_data["calcs_reversed"][0][
    #         "rot_entropy"
    #     ],
    #     "total_enthalpy_kcal/mol": total_enthalpy,
    #     "total_entropy_cal/molK": total_entropy,
    #     "translational_enthalpy_kcal/mol": recombination_data["calcs_reversed"][0][
    #         "trans_enthalpy"
    #     ],
    #     "translational_entropy_cal/molK": recombination_data["calcs_reversed"][0][
    #         "trans_entropy"
    #     ],
    #     "vibrational_enthalpy_kcal/mol": recombination_data["calcs_reversed"][0][
    #         "vib_enthalpy"
    #     ],
    #     "vibrational_entropy_cal/molK": recombination_data["calcs_reversed"][0][
    #         "vib_entropy"
    #     ],
    # }

    # data["vibration"] = {
    #     "IR_intensities": recombination_data["calcs_reversed"][0]["IR_intens"],
    #     "frequencies": recombination_data["calcs_reversed"][0]["frequencies"],
    #     "mode_vectors": recombination_data["calcs_reversed"][0][
    #         "frequency_mode_vectors"
    #     ],
    # }

    positions = molecule_graph.molecule.cart_coords.tolist()
    data["xyz"] = positions

    return data


def generate_molecule_ids(all_old_ids):
    """Generate new molecule ids."""
    # New molecule ids will be of the form "clibe-<6 digit number>"
    # Generate a random 6 digit number
    new_id = "mol-" + str(random.randint(100000, 999999))
    # Check if new_id is in all_old_ids
    if new_id in all_old_ids:
        # If it is, generate a new one
        new_id = generate_molecule_ids(all_old_ids)
    else:
        # If it is not, return it
        return new_id
