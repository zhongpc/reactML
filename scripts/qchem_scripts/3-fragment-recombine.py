from qcflow.crn_utils.fragmentation_recombination import get_all_molecule_from_json, FragmentReconnect
from qcflow.crn_utils.utils import instance_mongodb
import os
import yaml



# Get the database and create a collection to store the structures in
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

db = instance_mongodb(config)

# Create a collection to store the fragments
collection_name = config["workflow_tags"]["group"] + "_initial_graphs_collection"
initial_graphs_collection = getattr(db, collection_name) 


# Generate a molecule list
molecule_list = get_all_molecule_from_json('mol_fromFlattening_step2.json')


fragmenter = FragmentReconnect(
    initial_graphs_collection=initial_graphs_collection,
    molecule_list=molecule_list,
    depth= 1 ,
    bonding_factor_max=1.5,
    bonding_factor_min=1,
    bonding_factor_number=3,
    number_of_angles=100,
    debug= False, # args.debug,
)

fragmenter.run(if_add_monoatomic = False)
