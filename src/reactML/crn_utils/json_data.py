from typing import List, Iterable, Dict, Literal, Union
from collections import Counter
from datetime import datetime

import numpy as np

from reactML.crn_utils.fmt_convert import convert


def array_to_serializable(data):
    if isinstance(data, np.ndarray):
        serial_data = data.item() if data.size == 1 else data.tolist()
        return serial_data
    else:
        return data


class MoleculeRecord:
    """
    A data container for molecule information
    """
    def __init__(
        self,
        elements: List[str],
        coords: Iterable,
        smiles: str,
        inchikey: str,
        charge: int = 0,
        multiplicity: int = 1,
        note: str = None,
        date: str = None,
    ) -> None:
        """
        Args:
            smiles: SMILES string of the molecule
            inchikey: InChIKey of the molecule
            elements: list of element symbols
            coords: list of coordinates in Angstroms
        """
        self.smiles = smiles
        self.inchikey = inchikey
        self.elements = elements
        self.coords = array_to_serializable(coords)
        assert len(self.elements) == len(self.coords), "Number of elements must match number of coordinates"
        self.charge = charge
        self.multiplicity = multiplicity
        self.method_results: dict = None
        self.note = note
        self.date = date if date is not None else datetime.today().strftime("%Y-%m-%d")
    
    def __repr__(self) -> str:
        return f"MoleculeRecord(smiles={self.smiles}, inchikey={self.inchikey})"
    
    def __str__(self) -> str:
        return self.__repr__()

    def add_results(self, level_of_theory: str, results: Dict) -> None:
        """
        Add calculation results for a given level of theory

        Args:
            level_of_theory: a string representing the level of theory
            results: a dictionary of calculation results
        """
        if self.method_results is None:
            self.method_results = {}
        for key, value in results.items():
            results[key] = array_to_serializable(value)
        self.method_results[level_of_theory] = results

    def to_dict(self) -> Dict:
        """
        Convert the MoleculeRecord to a dictionary

        Returns:
            A dictionary representation of the MoleculeRecord
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "smiles": self.smiles,
            "inchikey": self.inchikey,
            "elements": self.elements,
            "coords": self.coords,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "method_results": self.method_results,
            "note": self.note,
            "date": self.date,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "MoleculeRecord":
        """
        Create a MoleculeRecord from a dictionary

        Args:
            d: a dictionary representation of a MoleculeRecord
        Returns:
            A MoleculeRecord instance
        """
        obj = cls(
            smiles=d["smiles"],
            inchikey=d["inchikey"],
            elements=d["elements"],
            coords=d["coords"],
            charge=d.get("charge", 0),
            multiplicity=d.get("multiplicity", 1),
            note=d.get("note", None),
            date=d.get("date", None),
        )
        obj.method_results = d.get("method_results", None)
        return obj

    def to_ase_atoms(self):
        """
        Convert the MoleculeRecord to an ASE Atoms object

        Returns:
            An ASE Atoms object representing the molecule
        """
        from ase import Atoms
        positions = np.array(self.coords)
        atoms = Atoms(symbols=self.elements, positions=positions)
        atoms.info["charge"] = self.charge
        # for MLIPs like MACE, UMA, ORB, etc.
        # this is not the same in PySCF, be super careful!
        # in PySCF, spin refers to the number of unpaired electrons, 2S
        # in these MLIPs, spin refers to the spin multiplicity, 2S+1
        atoms.info["spin"] = self.multiplicity
        return atoms

    def to_pymatgen_molecule(self):
        """
        Convert the MoleculeRecord to a Pymatgen Molecule object

        Returns:
            A Pymatgen Molecule object representing the molecule
        """
        from pymatgen.core import Molecule
        coords = np.array(self.coords)
        molecule = Molecule(
            species=self.elements,
            coords=coords,
            charge=self.charge,
            spin_multiplicity=self.multiplicity,
        )
        return molecule

    def is_same_molecule(self, other: "MoleculeRecord") -> bool:
        """
        Check if two MoleculeRecords represent the same molecule based on InChIKey

        Args:
            other: another MoleculeRecord to compare with
        Returns:
            True if both MoleculeRecords have the same InChIKey, False otherwise
        """
        return self.inchikey == other.inchikey


class ReactionRecord:
    """
    A data container for reaction information
    """
    def __init__(
        self,
        elements: List[str],
        reactant_coords: Iterable,
        product_coords: Iterable,
        reactant_smiles: str,
        product_smiles: str,
        reactant_inchikeys: Dict[str, int] = None,
        product_inchikeys: Dict[str, int] = None,
        charge: int = 0,
        multiplicity: int = 1,  # assume adiabatic PES
        note: str = None,
        date: str = None,
    ) -> None:
        """
        Args:
            elements: list of element symbols
            reactant_coords: list of reactant coordinates in Angstroms
            product_coords: list of product coordinates in Angstroms
            charge: overall charge of the reaction system
            multiplicity: overall spin multiplicity of the reaction system
        """
        self.elements = elements
        self.reactant_coords = array_to_serializable(reactant_coords)
        self.product_coords = array_to_serializable(product_coords)
        assert len(self.elements) == len(self.reactant_coords), "Number of elements must match number of reactant coordinates"
        assert len(self.elements) == len(self.product_coords), "Number of elements must match number of product coordinates"
        self.charge = charge
        self.multiplicity = multiplicity
        self.reactant_results: dict = None
        self.product_results: dict = None
        self.reaction_results: dict = None

        self.reactant_smiles = reactant_smiles
        self.product_smiles = product_smiles
        if reactant_inchikeys is None:
            react_smiles_list = reactant_smiles.split('.')
            self.reactant_inchikeys = dict(Counter([convert(smi, infmt='smiles', outfmt='inchikey', backend='rdkit') for smi in react_smiles_list]))
        else:
            self.reactant_inchikeys = reactant_inchikeys
        if product_inchikeys is None:
            prod_smiles_list = product_smiles.split('.')
            self.product_inchikeys = dict(Counter([convert(smi, infmt='smiles', outfmt='inchikey', backend='rdkit') for smi in prod_smiles_list]))
        else:
            self.product_inchikeys = product_inchikeys
        self.note = note
        self.date = date if date is not None else datetime.today().strftime("%Y-%m-%d")

    def add_results(
        self,
        level_of_theory: str,
        reactant_results: Dict = None,
        product_results: Dict = None,
        reaction_results: Dict = None,
    ) -> None:
        """
        Add calculation results for a given level of theory

        Args:
            level_of_theory: a string representing the level of theory
            reactant_results: a dictionary of reactant calculation results
            product_results: a dictionary of product calculation results
            reaction_results: a dictionary of reaction calculation results
        """
        if self.reactant_results is None:
            self.reactant_results = {}
        if self.product_results is None:
            self.product_results = {}
        if self.reaction_results is None:
            self.reaction_results = {}
        
        for key, value in reactant_results.items():
            reactant_results[key] = array_to_serializable(value)
        for key, value in product_results.items():
            product_results[key] = array_to_serializable(value)
        for key, value in reaction_results.items():
            reaction_results[key] = array_to_serializable(value)
        
        self.reactant_results[level_of_theory] = reactant_results
        self.product_results[level_of_theory] = product_results
        self.reaction_results[level_of_theory] = reaction_results
    
    def to_dict(self) -> Dict:
        """
        Convert the ReactionRecord to a dictionary

        Returns:
            A dictionary representation of the ReactionRecord
        """
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "elements": self.elements,
            "reactant_coords": self.reactant_coords,
            "product_coords": self.product_coords,
            "reactant_smiles": self.reactant_smiles,
            "product_smiles": self.product_smiles,
            "reactant_inchikeys": self.reactant_inchikeys,
            "product_inchikeys": self.product_inchikeys,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "reactant_results": self.reactant_results,
            "product_results": self.product_results,
            "reaction_results": self.reaction_results,
            "note": self.note,
            "date": self.date,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "ReactionRecord":
        """
        Create a ReactionRecord from a dictionary

        Args:
            d: a dictionary representation of a ReactionRecord
        Returns:
            A ReactionRecord instance
        """
        obj = cls(
            elements=d["elements"],
            reactant_coords=d["reactant_coords"],
            product_coords=d["product_coords"],
            reactant_smiles=d["reactant_smiles"],
            product_smiles=d["product_smiles"],
            reactant_inchikeys=d.get("reactant_inchikeys", None),
            product_inchikeys=d.get("product_inchikeys", None),
            charge=d.get("charge", 0),
            multiplicity=d.get("multiplicity", 1),
            note=d.get("note", None),
            date=d.get("date", None),
        )
        obj.reactant_results = d.get("reactant_results", None)
        obj.product_results = d.get("product_results", None)
        obj.reaction_results = d.get("reaction_results", None)
        return obj

    def _to_ase_atoms(self, coords: Iterable):
        from ase import Atoms
        positions = np.array(coords)
        atoms = Atoms(symbols=self.elements, positions=positions)
        atoms.info["charge"] = self.charge
        atoms.info["spin"] = self.multiplicity
        return atoms

    def reactants_to_ase_atoms(self):
        """
        Convert the reactants of the ReactionRecord to an ASE Atoms object

        Returns:
            An ASE Atoms object representing the reactants
        """
        return self._to_ase_atoms(self.reactant_coords)

    def products_to_ase_atoms(self):
        """
        Convert the products of the ReactionRecord to an ASE Atoms object

        Returns:
            An ASE Atoms object representing the products
        """
        return self._to_ase_atoms(self.product_coords)

    def _to_pymatgen_molecule(self, coords: Iterable):
        from pymatgen.core import Molecule
        coords = np.array(coords)
        molecule = Molecule(
            species=self.elements,
            coords=coords,
            charge=self.charge,
            spin_multiplicity=self.multiplicity,
        )
        return molecule

    def reactants_to_pymatgen_molecule(self):
        """
        Convert the reactants of the ReactionRecord to a Pymatgen Molecule object

        Returns:
            A Pymatgen Molecule object representing the reactants
        """
        return self._to_pymatgen_molecule(self.reactant_coords)
    
    def products_to_pymatgen_molecule(self):
        """
        Convert the products of the ReactionRecord to a Pymatgen Molecule object

        Returns:
            A Pymatgen Molecule object representing the products
        """
        return self._to_pymatgen_molecule(self.product_coords)

    def is_same_reaction(self, other: "ReactionRecord") -> bool:
        """
        Check if two ReactionRecords represent the same reaction based on reactant and product InChIKeys

        Args:
            other: another ReactionRecord to compare with
        Returns:
            True if both ReactionRecords have the same reactant and product InChIKeys, False otherwise
        """
        return (self.reactant_inchikeys == other.reactant_inchikeys) and (self.product_inchikeys == other.product_inchikeys)

    @property
    def reaction_smiles(self) -> str:
        """
        Generate a reaction SMILES string in the format "reactants>>products"

        Returns:
            A reaction SMILES string
        """
        return f"{self.reactant_smiles} -> {self.product_smiles}"
    
    def __repr__(self) -> str:
        return f"ReactionRecord({self.reaction_smiles})"
    
    def __str__(self) -> str:
        return self.__repr__()
