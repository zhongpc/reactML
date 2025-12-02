from typing import Literal, Dict, Callable

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem import rdDetermineBonds
    RDKit_AVAIL = True
except ImportError:
    RDKit_AVAIL = False
try:
    from openbabel import pybel
    OBABEL_AVAIL = True
except ImportError:
    OBABEL_AVAIL = False

# registries for conversion functions
from_registry: Dict[str, Callable] = {}
to_registry: Dict[str, Callable] = {}


def register_from(infmt: str):
    def decorator(func: Callable) -> Callable:
        if infmt in from_registry:
            raise ValueError(f"Conversion from {infmt} already registered.")
        from_registry[infmt] = func
        return func
    return decorator


def register_to(outfmt: str):
    def decorator(func: Callable) -> Callable:
        if outfmt in to_registry:
            raise ValueError(f"Conversion to {outfmt} already registered.")
        to_registry[outfmt] = func
        return func
    return decorator


@register_from("xyz")
def _from_xyz(xyz: str, backend: Literal["rdkit", "openbabel"], **kwargs):
    charge = kwargs.get("charge", 0)
    mutliplicity = kwargs.get("multiplicity", 1)
    if backend == "rdkit":
        rdkit_mol = Chem.MolFromXYZBlock(xyz)
        rdDetermineBonds.DetermineBonds(rdkit_mol, charge=charge)
        rdkit_mol = Chem.RemoveHs(rdkit_mol)
        return rdkit_mol
    elif backend == "openbabel":
        pybel_mol = pybel.readstring("xyz", xyz)
        pybel_mol.OBMol.SetTotalCharge(charge)
        pybel_mol.OBMol.SetTotalSpinMultiplicity(mutliplicity)
        return pybel_mol
    else:
        raise ValueError("Unsupported backend. Use 'rdkit' or 'openbabel'.")


@register_from("smiles")
def _from_smiles(smiles: str, backend: Literal["rdkit", "openbabel"], **kwargs):
    if backend == "rdkit":
        rdkit_mol = Chem.MolFromSmiles(smiles)
        return rdkit_mol
    elif backend == "openbabel":
        pybel_mol = pybel.readstring("smi", smiles)
        return pybel_mol
    else:
        raise ValueError("Unsupported backend. Use 'rdkit' or 'openbabel'.")


@register_from("inchi")
def _from_inchi(inchi: str, backend: Literal["rdkit", "openbabel"], **kwargs):
    if backend == "rdkit":
        rdkit_mol = Chem.MolFromInchi(inchi)
        return rdkit_mol
    elif backend == "openbabel":
        pybel_mol = pybel.readstring("inchi", inchi)
        return pybel_mol
    else:
        raise ValueError("Unsupported backend. Use 'rdkit' or 'openbabel'.")


@register_to("xyz")
def _to_xyz(core_mol, backend: Literal["rdkit", "openbabel"], **kwargs) -> str:
    if backend == "rdkit":
        assert isinstance(core_mol, Chem.Mol)
        core_mol = Chem.AddHs(core_mol)
        AllChem.EmbedMolecule(core_mol)
        AllChem.UFFOptimizeMolecule(core_mol)
        xyz = Chem.MolToXYZBlock(core_mol)
    elif backend == "openbabel":
        assert isinstance(core_mol, pybel.Molecule)
        core_mol.addh()
        core_mol.make3D()
        xyz = core_mol.write("xyz")
    else:
        raise ValueError("Unsupported backend. Use 'rdkit' or 'openbabel'.")
    return xyz


@register_to("smiles")
def _to_smiles(core_mol, backend: Literal["rdkit", "openbabel"], **kwargs) -> str:
    if backend == "rdkit":
        assert isinstance(core_mol, Chem.Mol)
        smiles = Chem.MolToSmiles(core_mol)
    elif backend == "openbabel":
        assert isinstance(core_mol, pybel.Molecule)
        smiles = core_mol.write("smi").strip()
    else:
        raise ValueError("Unsupported backend. Use 'rdkit' or 'openbabel'.")
    return smiles


@register_to("inchi")
def _to_inchi(core_mol, backend: Literal["rdkit", "openbabel"], **kwargs) -> str:
    if backend == "rdkit":
        assert isinstance(core_mol, Chem.Mol)
        inchi = Chem.MolToInchi(core_mol)
    elif backend == "openbabel":
        assert isinstance(core_mol, pybel.Molecule)
        inchi = core_mol.write("inchi").strip()
    else:
        raise ValueError("Unsupported backend. Use 'rdkit' or 'openbabel'.")
    return inchi


@register_to("inchikey")
def _to_inchikey(core_mol, backend: Literal["rdkit", "openbabel"], **kwargs) -> str:
    if backend == "rdkit":
        assert isinstance(core_mol, Chem.Mol)
        inchikey = Chem.MolToInchiKey(core_mol)
    elif backend == "openbabel":
        assert isinstance(core_mol, pybel.Molecule)
        inchikey = core_mol.write("inchikey").strip()
    else:
        raise ValueError("Unsupported backend. Use 'rdkit' or 'openbabel'.")
    return inchikey


def convert(
    in_data: str,
    infmt: str,
    outfmt: str,
    backend: Literal["rdkit", "openbabel"] = None,
    **kwargs,
) -> str:
    """
    Convert data from one format to another using registered conversion functions.

    Args:
        in_data: The input data as a string.
        infmt: The input format.
        outfmt: The output format.
        **kwargs: Additional keyword arguments for conversion functions.
    Returns:
        The converted data as a string.
    """
    if infmt not in from_registry:
        raise NotImplementedError(f"Input format {infmt} not supported.")
    if outfmt not in to_registry:
        raise NotImplementedError(f"Output format {outfmt} not supported.")
    # automatically handle backend if needed
    if backend is None:
        if RDKit_AVAIL:
            backend = "rdkit"
        elif OBABEL_AVAIL:
            backend = "openbabel"
        else:
            raise ImportError("No backend available for conversion.")
    else:
        if backend == "rdkit" and not RDKit_AVAIL:
            raise ImportError("RDKit backend requested but not available.")
        if backend == "openbabel" and not OBABEL_AVAIL:
            raise ImportError("OpenBabel backend requested but not available.")
    core_mol = from_registry[infmt](in_data, backend=backend, **kwargs)
    out_data = to_registry[outfmt](core_mol, backend=backend, **kwargs)
    return out_data


if __name__ == "__main__":
    test_smiles = "[Na+][O]=C1OCCO1"
    rdkit_xyz = convert(
        in_data=test_smiles,
        infmt="smiles",
        outfmt="xyz",
        backend="rdkit",
    )
    print(rdkit_xyz)
    rdkit_smiles = convert(
        in_data=rdkit_xyz,
        infmt="xyz",
        outfmt="smiles",
        backend="rdkit",
        charge=1,
    )
    print(rdkit_smiles)
