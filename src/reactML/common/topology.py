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
    try:  # use rdkit to parse xyz
        rdkit_mol = Chem.MolFromXYZBlock(xyz_str)
        rdDetermineBonds.DetermineBonds(rdkit_mol)
    except:  # use openbabel to parse xyz
        from openbabel import pybel
        # convert xyz string to sdf string
        pybel_mol = pybel.readstring("xyz", xyz_str)
        sdf_str = pybel_mol.write("sdf")
        # convert sdf string to rdkit mol
        rdkit_mol = Chem.MolFromMolBlock(sdf_str, sanitize=False, removeHs=False)
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
        P      -0.226615845101806      1.101045500529010      1.838347275521550
        F      -0.757400148995978      2.599747311280309      1.709453825294907
        F       0.411909839529254     -0.425302759247673      1.989450298417653
        F      -1.690834277636804      0.484450909757044      1.695093923199597
        F      -0.376160913208060      1.155914905887633      3.436973498718262
        F       1.330298410281703      1.656559699579542      2.003825461677867
        O      -0.066247400632948      1.041706954284057      0.153485262615070
        O       0.751539726396582      0.693593208840418     -1.860377911358127
        C       0.923841015496014      0.609857145107110     -0.628201012218462
        O       2.005440080608129      0.129584142765855     -0.145710916291551
        Li      2.487677478734471     -0.074568743072396     -2.009812614959682
        Li      2.278642034529441     -0.003998275710907      1.663602909382929
    """
    pyscf_mol.basis = "sto-3g"
    pyscf_mol.build()

    rdkit_mol = rdkit_mol_from_pyscf(pyscf_mol)
    print(get_constraints_idx(rdkit_mol))