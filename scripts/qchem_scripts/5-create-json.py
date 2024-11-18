from json import loads
from typing import List
import copy
import logging
import yaml
import numpy as np
import re

from qcflow.crn_utils.utils import instance_mongodb, read_db_json
from qcflow.crn_utils.qrrho import QuasiRRHO
from qcflow.crn_utils.rrho_shifted import ShiftedRRHO
from qcflow.crn_utils.mrnet_utils import remove_high_energy_mol_entries

from monty.serialization import dumpfn, loadfn
from pymongo import UpdateOne
from pymongo.errors import OperationFailure
from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.symmetry.analyzer import PointGroupAnalyzer
from pymatgen.analysis.local_env import OpenBabelNN
from atomate.qchem.database import QChemCalcDb
from mrnet.core.mol_entry import MoleculeEntry
from ase import units as ase_units


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig()

kcal_per_mol_to_eV = 0.0433641



#### Get the database and create a collection to store the structures in
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)
db = instance_mongodb(config)
db_config_json = read_db_json(config["db_file"])



entries = list()
collection_NAME = config["workflow_tags"]["group"] + "_fragmentation_recombination"

try:
    db[collection_NAME].create_index("molecule_id", unique=True)
    # db["Mg_fragmentation_recombination"].create_index("molecule_id", unique=True)
except OperationFailure:
    print("Index already existed")

print("LOADING")


########### Load the molecule documents and quantities ############

collection = getattr(db, db_config_json["collection"])

tags_recombination_freq_opt = {"tags.group": config["workflow_tags"]["group"] + "_fragmentation_recombination",  # should be "Li_SEI_fragmentation_recombination_quantity"
                               "task_label": "frequency_flattening_opt"}
                                 

recombination_freq_opt_collection = collection.find(tags_recombination_freq_opt)



mol_docs = list(recombination_freq_opt_collection)
# mol_docs = list(db["Mg_fragmentation_recombination"].find())
# mol_quant = list(db["Mg_fragmentation_recombination_quantities"].find())
print("Total number of mol_doc (before cleaning): ", len(mol_docs))


mol_documents = [] # representing the frequency_flattening_opt
mol_quantities = [] #  representing the single point calculations

collection = getattr(db, db_config_json["collection"])

for idx, mol_doc in enumerate(mol_docs):
    mol_doc

    # Find the corresponding recombination_data_quantities
    tags_moldoc = mol_doc["tags"]
    tags_mol_quant_ = copy.deepcopy(tags_moldoc)

    # print(tags_mol_quant_)
    # tags_mol_quant_["task_label"] = "single_point"


    smiles_tag = mol_doc["smiles"]
    formula_pretty_tag = mol_doc["formula_pretty"]
    formula_alph_tag = mol_doc["formula_alphabetical"]
    # basis_tag = mol_doc["calcs_reversed"][0]["input"]["rem"]["basis"]
    # method_tag = mol_doc["calcs_reversed"][0]["input"]["rem"]["method"]
    optimized_molecule_tag = mol_doc["output"]['initial_molecule'] # for single point calculations, mol_doc["output"]["optimized_molecule"]
    # task_label = optimized_data["task_label"]
    # molecule_name = task_label[:-4] + "_" + basis_tag + "_" + method_tag

    tags_mol_quantities = {"tags": tags_mol_quant_, 
                           "task_label": "single_point",}
                # "smiles":smiles_tag, "formula_pretty":formula_pretty_tag,
                # "formula_alphabetical":formula_alph_tag,
                # # "calcs_reversed.0.input.rem.basis":basis_tag,
                # # "calcs_reversed.0.input.rem.method":method_tag,
                # "input.initial_molecule":optimized_molecule_tag,
    
    quantities_collection = collection.find_one(tags_mol_quantities)

    if quantities_collection is not None:
        mol_quantities.append(quantities_collection)
        mol_documents.append(mol_doc)
    else:
        logger.warning(
            f"Skipping {idx} because it has no quantities, compute single point first."
        )


print("Total number of mol_documents (representing freq_flattening_opt): ", len(mol_documents))
print("Total number of mol_documents (representing single_point): ",len(mol_quantities))



#### clean the mol_docs, make sure each doc has both frequency_flattening_opt and single point calculations
print("ClEAN THE MOL_DOCS")
mol_docs = []
for document, quantity in zip(mol_documents, mol_quantities): # document is for frequency_flattening_opt, qunatity is for single point
    mol_docs.append([document, quantity])



print("CREATING MOL ENTRIES")
mol_entries = list()
for doc in mol_docs:
    task_id = doc[0]["task_id"]

    output = copy.deepcopy(doc[0]["output"])
    output['molecule'] = output['initial_molecule'] # read from the single point calculation
    output['energy_Ha'] = output['final_energy']
    output['enthalpy_kcal/mol'] = None # output['enthalpy']
    output['entropy_cal/molK'] = None # output['entropy']
    output['task_id'] = task_id
    
    
    mol_entries.append(MoleculeEntry.from_molecule_document(output, task_id))



print(len(mol_entries))
print("FILTERING -- removing high energy molecules")
mol_entries_filtered = remove_high_energy_mol_entries(mol_entries)
good_task_ids = [entry.entry_id for entry in mol_entries_filtered]
mol_docs = [doc for doc in mol_docs if doc[0]["task_id"] in good_task_ids]
print("Total number of mol_entries_filtered: ", len(mol_entries_filtered))
print("Total number of good_task_ids: ", len(good_task_ids))
print("Total number of mol_docs (after cleaning): ", len(mol_docs))



ignore_nbo = False # only for testing

print("BUILDING")
operations = list()
for mol_doc in mol_docs:
    entry = dict()
    # keys_list = list(mol_doc[0].keys())
    # print(keys_list)
    # keys_list2 = list(mol_doc[1].keys())
    # print(keys_list2)
    # print(mol_doc[])
    # print("\n")

    # # For now, don't allow any incomplete entry to be included
    # if ("bonds" not in mol_doc or "vibrational_frequencies" not in mol_doc) and len(mol_doc["molecule"]["sites"]) > 1:
    #     print("NO BONDS FOR MOLECULE WITH MORE THAN 1 ATOM: {}".format(mol_doc["task_id"]))
    #     continue

    # For now, don't allow any incomplete entry to be included
    if ("frequencies" not in mol_doc[0]["calcs_reversed"][0]) and len(mol_doc["molecule"]["sites"]) > 1:
        print("NO BONDS FOR MOLECULE WITH MORE THAN 1 ATOM: {}".format(mol_doc["task_id"]))
        continue
    

    
    # # For now, don't allow any incomplete entry to be included
    # if (len(mol_doc[0]["output"]["initial_molecule"]["sites"])) == 1:
    #     print("NO BONDS FOR MOLECULE WITH MORE THAN 1 ATOM: {}".format(mol_doc["task_id"]))
    #     continue

    # Molecule ID
    mol_id = "MolID-" + str(mol_doc[0]["task_id"])
    entry["molecule_id"] = mol_id
    # print(entry)

    # Basic molecule information
    mol = Molecule.from_dict(mol_doc[0]["output"]["optimized_molecule"])
    # print(mol.keys())
    species = [str(e) for e in mol.species]
    coords = mol.cart_coords.tolist()
    entry["molecule"] = mol_doc[0]["output"]["optimized_molecule"]
    entry["charge"] = mol.charge
    entry["spin_multiplicity"] = mol.spin_multiplicity
    entry["species"] = species
    entry["xyz"] = coords
    # entry["bonds"] = mol_doc[1]["output"]["initial_molecule"].get("bonds", {"critic": list(), "nbo": list()})
    # entry["molecule_graph"] = MoleculeGraph.with_edges(
    #     mol,
    #     {tuple(i): {} for i in entry["bonds"]["critic"]}
    # ).as_dict()
    # print(entry["bonds"])

    molecule = mol_doc[0]["output"]["optimized_molecule"]
    molecule = Molecule.from_dict(molecule)
    molecule_graph = MoleculeGraph.with_local_env_strategy(molecule, OpenBabelNN())
    entry["molecule_graph"] = molecule_graph

    entry["number_atoms"] = len(mol_doc[0]["output"]["optimized_molecule"]["sites"])
    entry["number_elements"] = len(molecule_graph.molecule.composition.elements)
    entry["composition"] = molecule_graph.molecule.composition
    entry["formula_alphabetical"] = mol_doc[0]["formula_alphabetical"]
    entry["chemical_system"] = mol_doc[0]["chemsys"]

    elements_counts = re.findall(r'([A-Z][a-z]*)(\d*)', entry["formula_alphabetical"])
    composition = {}
    elements = []

    for element, count in elements_counts:
        if count:
            composition[element] = float(count)
        else:
            composition[element] = 1.0  # If no count is specified, assume 1
        elements.append(element)

    entry["composition"] = composition
    entry["elements"] = elements

    # Atomic partial charges
    entry["partial_charges"] = dict()
    entry["partial_charges"]["resp"] = list(mol_doc[0]["output"]["resp"])
    if mol.spin_multiplicity == 1:
        entry["partial_charges"]["mulliken"] = list(mol_doc[0]["output"]["mulliken"])
    else:
        entry["partial_charges"]["mulliken"] = [x[0] for x in mol_doc[0]["output"]["mulliken"]]
    if "critic" in mol_doc:
        entry["partial_charges"]["critic2"] = list(mol_doc["critic"]["processed"]["charges"])
    else:
        entry["partial_charges"]["critic2"] = None

    if ignore_nbo:
        pass
    else:
        if "nbo" in mol_doc[1]["output"]:
            entry["partial_charges"]["nbo"] = [float(mol_doc[1]["output"]["nbo"]["natural_populations"][0]["Charge"][str(i)]) for i in range(len(Molecule.from_dict(mol_doc[1]["output"]["initial_molecule"])))]
        
        else:
            entry["partial_charges"]["nbo"] = None

        if mol.spin_multiplicity == 1:
            entry["partial_spins"] = {"mulliken": None, "nbo": None}
        else:
            entry["partial_spins"] = {"mulliken": [x[1] for x in mol_doc[1]["output"]["mulliken"]],
                                        "nbo": [float(mol_doc[1]["output"]["nbo"]["natural_populations"][0]["Density"][str(i)]) for i in range(len(Molecule.from_dict(mol_doc[1]["output"]["initial_molecule"])))]}

        nbo_warnings = mol_doc[0]["warnings"]

    # Vibrational information
    if len(mol) > 1:
        entry["vibration"] = dict()
        entry["vibration"]["frequencies"] = list(mol_doc[0]["calcs_reversed"][0]["frequencies"])
        entry["vibration"]["mode_vectors"] = list(mol_doc[0]["calcs_reversed"][0]["frequency_mode_vectors"])
        entry["vibration"]["IR_intensities"] = list(mol_doc[0]["calcs_reversed"][0]["IR_intens"])
    else:
        # Doesn't make sense for single atoms
        entry["vibration"] = None

    # Molecular thermodynamics
    entry["thermo"] = {"raw": dict(),
                        "eV": dict(),
                        "quasi_rrho_eV": dict(),
                        "shifted_rrho_eV": dict()}

    entry["thermo"]["raw"]["electronic_energy_Ha"] = mol_doc[0]["output"]["final_energy"]
    entry["thermo"]["raw"]["total_enthalpy_kcal/mol"] = mol_doc[0]["calcs_reversed"][0]["total_enthalpy"]
    entry["thermo"]["raw"]["total_entropy_cal/molK"] = mol_doc[0]["calcs_reversed"][0]["total_entropy"]
    if len(mol) > 1:
        entry["thermo"]["raw"]["translational_enthalpy_kcal/mol"] = mol_doc[0]["calcs_reversed"][0]["trans_enthalpy"]
        entry["thermo"]["raw"]["rotational_enthalpy_kcal/mol"] = mol_doc[0]["calcs_reversed"][0]["rot_enthalpy"]
        entry["thermo"]["raw"]["vibrational_enthalpy_kcal/mol"] = mol_doc[0]["calcs_reversed"][0]["vib_enthalpy"]
        entry["thermo"]["raw"]["translational_entropy_cal/molK"] = mol_doc[0]["calcs_reversed"][0]["trans_entropy"]
        entry["thermo"]["raw"]["rotational_entropy_cal/molK"] = mol_doc[0]["calcs_reversed"][0]["rot_entropy"]
        entry["thermo"]["raw"]["vibrational_entropy_cal/molK"] = mol_doc[0]["calcs_reversed"][0]["vib_entropy"]
    else:
        entry["thermo"]["raw"]["translational_enthalpy_kcal/mol"] = mol_doc[0]["calcs_reversed"][0]["total_enthalpy"]
        entry["thermo"]["raw"]["rotational_enthalpy_kcal/mol"] = None
        entry["thermo"]["raw"]["vibrational_enthalpy_kcal/mol"] = None
        entry["thermo"]["raw"]["translational_entropy_cal/molK"] = mol_doc[0]["calcs_reversed"][0]["total_entropy"]
        entry["thermo"]["raw"]["rotational_entropy_cal/molK"] = None
        entry["thermo"]["raw"]["vibrational_entropy_cal/molK"] = None

    entry["thermo"]["eV"]["electronic_energy"] = entry["thermo"]["raw"]["electronic_energy_Ha"] * ase_units.Hartree
    entry["thermo"]["eV"]["total_enthalpy"] = entry["thermo"]["raw"]["total_enthalpy_kcal/mol"] * kcal_per_mol_to_eV
    entry["thermo"]["eV"]["total_entropy"] = entry["thermo"]["raw"]["total_entropy_cal/molK"] * kcal_per_mol_to_eV / 1000
    entry["thermo"]["eV"]["free_energy"] = entry["thermo"]["eV"]["electronic_energy"] + entry["thermo"]["eV"]["total_enthalpy"] - 298.15 * entry["thermo"]["eV"]["total_entropy"]

    pga = PointGroupAnalyzer(mol)
    entry["point_group"] = pga.sch_symbol

    rotational_symmetry_numbers = {1: ["C1", "Cs", "Ci", "C*v", "S2"],
                                    2: ["C2", "C2h", "C2v", "S4", "D*h"],
                                    3: ["C3", "C3h", "C3v", "S6"],
                                    4: ["C4v", "D4h", "D4d", "D2", "D2h", "D2d"],
                                    5: ["C5v", "Ih"],
                                    6: ["D3", "D3h", "D3d"],
                                    10: ["D5h", "D5d"],
                                    12: ["T", "Td", "Th", "D6h"],
                                    14: ["D7h"],
                                    16: ["D8h"],
                                    24: ["Oh"],
                                    np.inf: ["Kh"]}

    # Skip for single atoms
    if len(mol) > 1:
        r = 1
        for rot_num, point_groups in rotational_symmetry_numbers.items():
            if entry["point_group"] in point_groups:
                r = rot_num
                break
        if entry["point_group"] in ["C*v", "D*h"]:
            linear = True
        else:
            linear = False

        rrho_input = {"mol": mol,
                        "mult": mol.spin_multiplicity,
                        "frequencies": entry["vibration"]["frequencies"],
                        "elec_energy": entry["thermo"]["raw"]["electronic_energy_Ha"]}

        try:
            qrrho = QuasiRRHO(rrho_input, sigma_r=r, linear=linear)
            entry["thermo"]["quasi_rrho_eV"]["electronic_energy"] = entry["thermo"]["eV"]["electronic_energy"]
            entry["thermo"]["quasi_rrho_eV"]["total_enthalpy"] = qrrho.enthalpy_quasiRRHO * 27.2114
            entry["thermo"]["quasi_rrho_eV"]["total_entropy"] = qrrho.entropy_quasiRRHO * 27.2114
            entry["thermo"]["quasi_rrho_eV"]["free_energy"] = qrrho.free_energy_quasiRRHO * 27.2114
        except ZeroDivisionError:
            entry["thermo"]["quasi_rrho_eV"] = None

        try:
            rrho_shifted = ShiftedRRHO(rrho_input, sigma_r=r, linear=linear)
            entry["thermo"]["shifted_rrho_eV"]["electronic_energy"] = entry["thermo"]["eV"]["electronic_energy"]
            entry["thermo"]["shifted_rrho_eV"]["total_enthalpy"] = rrho_shifted.enthalpy_RRHO * 27.2114
            entry["thermo"]["shifted_rrho_eV"]["total_entropy"] = rrho_shifted.entropy_RRHO * 27.2114
            entry["thermo"]["shifted_rrho_eV"]["free_energy"] = rrho_shifted.free_energy_RRHO * 27.2114
        except ZeroDivisionError:
            entry["thermo"]["shifted_rrho_eV"] = None
    else:
        entry["thermo"]["quasi_rrho_eV"] = None
        entry["thermo"]["shifted_rrho_eV"] = None

    entry["redox"] = {"electron_affinity_eV": None,
                        "ionization_energy_eV": None}
    # if mol_doc[0]["vertical_ea"] is not None:
    #     entry["redox"]["electron_affinity_eV"] = mol_doc["vertical_ea"] * 27.2114
    # if mol_doc[0]["vertical_ie"] is not None:
    #     entry["redox"]["ionization_energy_eV"] = mol_doc["vertical_ie"] * 27.2114

    entries.append(entry)



########### save the entries for CRN analysis ############
save_filename = "entries_" + config["workflow_tags"]["group"] + "_step5.json"
dumpfn(entries, save_filename)


cleaned_entries = loads(open(save_filename).read())

for entry in cleaned_entries:
    operations.append(
        UpdateOne(
            {"molecule_id": entry["molecule_id"]},
            {"$set": entry},
            upsert=True
        )
    )

print("WRITING {} ENTRIES".format(len(operations)))
# db.db["tasks"].bulk_write(operations)
print("COMPLETED")