import os
import copy
import yaml
import datetime

from pymatgen.core.structure import Molecule
from fireworks import Workflow, LaunchPad
from atomate.qchem.database import QChemCalcDb
from atomate.qchem.fireworks.core import FrequencyFlatteningOptimizeFW, SinglePointFW
from atomate.common.powerups import add_tags

def add_mol_to_workflow(folder_path, charge, config):
    """
    Add molecules to the workflow within one folder.
    Each folder contains the molecules with the same charge.
    """

    input_params = config['input_params_step1']
    qdb = QChemCalcDb.from_db_file(config['db_file'])
    lp = LaunchPad.from_file(config['launchpad_file'])

    for filename in os.listdir(folder_path):
        if filename.endswith(".xyz"):
            file_path = os.path.join(folder_path, filename)

            mol = Molecule.from_file(file_path)
            nelectrons_neutral = mol.nelectrons

            nelectrons = nelectrons_neutral - float(charge)
            spin_multiplicity = 1 if nelectrons % 2 == 0 else 2

            mol.set_charge_and_spin(charge=float(charge), spin_multiplicity=spin_multiplicity)

            if mol.num_sites == 1:
                fw = SinglePointFW(
                    molecule=mol,
                    max_cores=">>max_cores<<",
                    qchem_input_params=input_params,
                    name="single_point",
                    qchem_cmd=">>qchem_cmd<<",
                    db_file=">>db_file<<",
                    )

                file_name = filename[:-4]  # remove the xyz extension
                config['workflow_tags']['date'] = datetime.datetime.now().strftime("%Y-%m-%d")

                wf = Workflow([fw], name=file_name)
                wf = add_tags(wf, config['workflow_tags'])
                
                print(wf)
                lp.add_wf(wf)

            else:
                fw_1 = FrequencyFlatteningOptimizeFW(
                    molecule=mol,
                    name="frequency_flattening_opt",
                    qchem_input_params=copy.deepcopy(input_params),
                    db_file=">>db_file<<"
                )

                fw_2 = SinglePointFW(
                    max_cores=">>max_cores<<",
                    qchem_input_params=input_params,
                    name="single_point",
                    qchem_cmd=">>qchem_cmd<<",
                    db_file=">>db_file<<",
                    parents = fw_1,
                    )

                file_name = filename[:-4]  # remove the xyz extension
                config['workflow_tags']['date'] = datetime.datetime.now().strftime("%Y-%m-%d")

                wf = Workflow([fw_1, fw_2], name=file_name)
                wf = add_tags(wf, config['workflow_tags'])
                
                print(wf)
                lp.add_wf(wf)


# Read the YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


# Set the folder path and charge state
folder_path_list = ["./init_mols/"]
charge_list = [0]  # set the initial charge state to zero

for folder_path, charge in zip(folder_path_list, charge_list):
    add_mol_to_workflow(folder_path, charge, config)
