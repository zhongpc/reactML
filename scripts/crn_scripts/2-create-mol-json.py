from datetime import datetime
import logging
import yaml

from monty.serialization import loadfn, dumpfn

from qcflow.crn_utils.utils import get_required_schema, generate_molecule_ids, instance_mongodb, read_db_json

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN




kcal_per_mol_to_eV = 0.0433641

#### get the information from the lpad ####
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()


#### read the .yaml config file####
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

db = instance_mongodb(config)


### Loop to check the current datat ###
current_data = []
# Store all of the current molecules in a list
#  list_current_graphs, list_ids = get_all_graphs(current_data)
list_current_graphs, list_ids = [],[]

# Get the new structures from the database
db_config_json = read_db_json(config["db_file"])
collection = getattr(db, db_config_json["collection"])  # db.qchem

recombination_collection = collection.find(
            {
                "tags.group": config["workflow_tags"]["group"],
                "tags.class": config["workflow_tags"]["class"],
            }
        )


print(recombination_collection)

for idx, recombination_data in enumerate(recombination_collection):
    print(recombination_data)

    if recombination_data['task_label'] == 'single_point':
        pass
    else:
        continue

    # Based on if the structure changes or not, write out the fragments
    # to a separate directory
    structure_change = recombination_data["structure_change"]
    # Allow only unique string entries in structure_change
    structure_change = list(set(structure_change))

    # if "unconnected_fragments" in structure_change:
    #     logger.warning(f"Skipping {idx} because it is unconnected")
    #     continue

    # tags_recombination = recombination_data["tags"]
    # tags_recombination_quantities_ = copy.deepcopy(tags_recombination)
    # tags_recombination_quantities_[
    #     "group"
    # ] = "Mg_fragmentation_recombination_quantities"
    
    tags_recombination_quantities = {
                                     "tags.group": config["workflow_tags"]["group"],
                                     "tags.class": config["workflow_tags"]["class"],
                                     }

    # Find the corresponding recombination_data_quantities
    recombination_quantities_collection = collection.find_one(
        tags_recombination_quantities
    )

    print(recombination_quantities_collection)

    if recombination_quantities_collection == None:
            logger.warning(
                f"Skipping {idx} because it has no quantities, compute single points first."
            )
            continue
    
    

    # Get the molecule and molecule graph
    molecule = recombination_data["output"]["initial_molecule"]
    molecule = Molecule.from_dict(molecule)
    molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())
    
    # Check if the molecule graph is already in the list
    skip_idx = False
    for check_graph in list_current_graphs:
        if molecule_graph.isomorphic_to(check_graph):
            if molecule_graph.molecule.charge == check_graph.molecule.charge:
                if (
                    molecule_graph.molecule.spin_multiplicity
                    == check_graph.molecule.spin_multiplicity
                ):
                    logger.warning(
                        f"Skipping {idx} because it is already in the list"
                    )
                    skip_idx = True
                    break
    if skip_idx:
        continue

    # At this point, we have a new molecule graph
    # which is unique and can be added to the list
    molecule_id = generate_molecule_ids(list_ids)
    logger.info(f"Adding {idx} with id {molecule_id}")
    assert molecule_id is not None

    recombination_data_list = [
        recombination_data,
        recombination_quantities_collection,
    ]
    schema_add_ = get_required_schema(
        molecule_graph, recombination_data_list, molecule_id=molecule_id
    )
    logger.info(f"Adding {idx} with id {molecule_id}")

    # Add the new molecule to the list
    current_data.append(schema_add_)


# Write out the new mol_json file
# Create a string with the date and time of creating the file
dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
# Use `dt_string` to create a new file name
new_file_name = "mol_fromFlattening_step2.json"

# Write out the new file
print(len(current_data))
dumpfn(current_data, new_file_name)


for item in current_data:
    mol = Molecule.from_dict(item['molecule'])
    print(len(mol))
