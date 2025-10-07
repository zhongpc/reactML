from typing import List, Tuple

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from pyscf import gto


def rdkit_mol_from_pyscf(pyscf_mol: gto.Mole) -> Chem.Mol:
    """Convert a PySCF molecule to an RDKit molecule.

    Args:
        pyscf_mol: PySCF molecule object.

    Returns:
        RDKit molecule object.
    """
    xyz_str = f"{pyscf_mol.natm}\n\n"
    for i in range(pyscf_mol.natm):
        atom = pyscf_mol.atom_symbol(i)
        x, y, z = pyscf_mol.atom_coord(i, unit="Angstrom")
        xyz_str += f"{atom} {x:.6f} {y:.6f} {z:.6f}\n"
    rdkit_mol = Chem.MolFromXYZBlock(xyz_str)
    # add topology
    rdDetermineBonds.DetermineBonds(rdkit_mol)
    return rdkit_mol


def get_constraints_idx(rdkit_mol: Chem.Mol) ->  Tuple[List[int], List[List[int]]]:
    """
    1. Get the indices for sum constraints during the second stage of RESP fitting.
    i.e., add the constraint to those atoms that not SP3/SP2 carbon and not hydrogen connected to these carbons.

    2. Get the indices for equal constraints during the second stage of RESP fitting.
    i.e., add the constraint to those atoms that are chemically equivalent.

    Args:
        rdkit_mol: RDKit molecule object.
    Returns:
        A tuple containing:
        - List of indices for sum constraints.
        - List of lists of indices for equal constraints.
    """
    # step 1: sum constraints
    fitted_carbon_idx = set()
    fitted_hydrogen_idx = set()
    for idx, atom in enumerate(rdkit_mol.GetAtoms()):
        atom: Chem.Atom
        if atom.GetSymbol() == "C":
            # case 1: SP3 carbon and its bonded hydrogens
            if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
                fitted_carbon_idx.add(idx)
                for neighbor in atom.GetNeighbors():
                    neighbor: Chem.Atom
                    if neighbor.GetSymbol() == "H":
                        fitted_hydrogen_idx.add(neighbor.GetIdx())
            # case 2: =CH2 and its bonded hydrogens
            if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                neighbor_h_idx = [neighbor.GetIdx() for neighbor in atom.GetNeighbors() if neighbor.GetSymbol() == "H"]
                if len(neighbor_h_idx) == 2:
                    fitted_carbon_idx.add(idx)
                    fitted_hydrogen_idx.update(neighbor_h_idx)
    all_indices = set(range(rdkit_mol.GetNumAtoms()))
    sum_constraints = list(all_indices - fitted_hydrogen_idx - fitted_carbon_idx)

    # step 2: equal constraints
    symm_classes = Chem.CanonicalRankAtoms(rdkit_mol, breakTies=False, includeChirality=False)
    symm_groups = {}
    for idx, symm_class in enumerate(symm_classes):
        if idx not in fitted_hydrogen_idx:
            continue
        if symm_class not in symm_groups:
            symm_groups[symm_class] = []
        symm_groups[symm_class].append(idx)
    equal_constraints = [group for group in symm_groups.values() if len(group) > 1]

    return sorted(sum_constraints), equal_constraints


if __name__ == "__main__":
    # Example usage
    pyscf_mol = gto.Mole()
    pyscf_mol.atom = """
  C    0.0000000   -0.0000000   -0.7218894
  O    0.0000000    1.1191118    0.0073889
  C   -0.0000000    0.7709575    1.3372793
  C   -0.0000000   -0.7709575    1.3372793
  O   -0.0000000   -1.1191118    0.0073889
  O    0.0000000   -0.0000000   -1.9416719
  H   -0.9123465    1.1712556    1.8293877
  H    0.9123465    1.1712556    1.8293877
  H   -0.9123465   -1.1712556    1.8293877
  H    0.9123465   -1.1712556    1.8293877"""
    pyscf_mol.basis = "sto-3g"
    pyscf_mol.build()

    rdkit_mol = rdkit_mol_from_pyscf(pyscf_mol)
    print(get_constraints_idx(rdkit_mol))