import os
import yaml
import logging
from atomate.qchem.fireworks.core import FrequencyFlatteningOptimizeFW
from atomate.qchem.fireworks.core import SinglePointFW
from atomate.common.powerups import add_tags
from fireworks import LaunchPad, Workflow
from pymatgen.analysis.graphs import MoleculeGraph

from qcflow.crn_utils.utils import get_all_graphs_from_collection, check_already_completed, instance_mongodb, read_db_json



# Get the database and create a collection to store the structures in
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

db_config_json = read_db_json(config["db_file"])

input_params = config['input_params_step4']


lp = LaunchPad.from_file(config["launchpad_file"])
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()


# Get the database and create a collection to store the structures in
db = instance_mongodb(config)
# get the collection from database
collection = getattr(db, db_config_json["collection"])  # db.qchem
collection_initial_structures =  getattr(db, config["workflow_tags"]["group"] + "_initial_graphs_collection") # graph collection with pymatgen class



# Get the fragments of the molecules
recombination_collection = collection_initial_structures.find({
    "tags.group": "initial_structures_fragmentation_recombination",})


for ii, entry in enumerate(recombination_collection):
    # print(ii)
    molecule_graph = MoleculeGraph.from_dict(entry)
    molecule = molecule_graph.molecule

    if len(molecule) == 1:
        print(ii)
    # print(molecule)

# Get older structures that have already been calculated.

collection_name = config["workflow_tags"]["group"] + "_recombination"
calculated_collection = collection.find(
    {
        # "tags.type_structure": "fragment",
        "tags.group": collection_name,
    }
)
all_graphs = get_all_graphs_from_collection(calculated_collection)
print(len(all_graphs))

NAME = "recombination_calculation"


output_dir = os.path.join("outputs", "new_recombination_structures")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

count_structures = 0
accepted_structures = 0


# Get the fragments of the molecules
recombination_collection = collection_initial_structures.find({
    "tags.group": "initial_structures_fragmentation_recombination",
})


# Write out the recombination structures all in separate xyz files.
for idx, recombination_data in enumerate(recombination_collection):
    print(idx)
    # tags_recombination = recombination_data["tags"]
    tags_recombination = recombination_data.pop("tags") # why use pop ?

    tags_recombination["group"] = config["workflow_tags"]["group"] + "_fragmentation_recombination" # "Li_SEI_fragmentation_recombination"

    molecule_graph = MoleculeGraph.from_dict(recombination_data)

    # Increment the count
    count_structures += 1

    # Check if the molecule graph is already in the list
    if check_already_completed(molecule_graph, all_graphs):
        logger.warning(f"Skipping {idx} because it is already in the list")
        continue
    molecule = molecule_graph.molecule
    accepted_structures += 1

    print("Accepted molecule: ", molecule.formula)

    if len(molecule) == 1:  # for single atom molecules
        firew = SinglePointFW(
            molecule=molecule,
            qchem_input_params=input_params,
            name="single_point",
            db_file=">>db_file<<",
        )
        # Create a workflow for just one simple firework
        wf = Workflow([firew], name=NAME)
        wf = add_tags(wf, tags_recombination)

    else:
        fw_1 = FrequencyFlatteningOptimizeFW(
            molecule=molecule,
            name="frequency_flattening_opt",
            qchem_input_params=input_params,
            db_file=">>db_file<<"
        )

        fw_2 = SinglePointFW(
            max_cores=">>max_cores<<",
            qchem_input_params=input_params,
            name="single_point",
            qchem_cmd=">>qchem_cmd<<",
            db_file=">>db_file<<",
            parents = fw_1,)
        
        
        wf = Workflow([fw_1, fw_2], name=NAME)
        wf = add_tags(wf, tags_recombination)
    
    lp.add_wf(wf)

logger.info(f"{count_structures} structures processed for calculation.")
logger.info(f"{accepted_structures} structures accepted for calculation.")
