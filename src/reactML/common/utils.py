import os.path as osp
from pyscf import gto
try:
    from gpu4pyscf import dft
except:
    from pyscf import dft


def write_xyz(mol: gto.Mole, filename: str):
    elements = mol.elements
    coords = mol.atom_coords(unit="Angstrom")
    with open(filename, "w") as f:
        f.write(f"{len(elements)}\n")
        # Convert spin (2S) to multiplicity (2S+1) format, as required by quantum chemistry conventions.
        f.write(f"{mol.charge} {mol.spin + 1}\n")
        for ele, coord in zip(elements, coords):
            f.write(f"{ele: <2}    {coord[0]:10.6f}    {coord[1]:10.6f}    {coord[2]:10.6f}\n")


def read_pcm_eps():
    # from https://gaussian.com/scrf/
    pcm_eps_txt = osp.join(osp.dirname(__file__), "pcm_eps.txt")
    with open(pcm_eps_txt, "r") as f:
        lines = f.readlines()
    eps_dict = {}
    for line in lines:
        solvent, eps = line.split(": ")
        eps_dict[solvent.strip().lower()] = float(eps.strip())
    return eps_dict


def build_dft(mol: gto.Mole, **kwargs):
    """
    Build a PySCF mean field object with the given molecule and parameters.
    Args:
        mol (gto.Mole): The molecule object.
        kwargs: Additional parameters for the mean field object.
    Returns:
        mf (pyscf.dft.KS): The mean field object.
    """
    # convert memory from GB to MB
    max_memory = kwargs.get("max_memory", None)
    if max_memory is not None:
        max_memory *= 1024  # convert GB to MB

    # build mole
    basis = kwargs.get("basis", "def2-SVP")
    charge = kwargs.get("charge", 0)
    spin = kwargs.get("spin", 0)
    mol.build(basis=basis, charge=charge, spin=spin, max_memory=max_memory)
    
    # set exchange-correlation functional
    xc = kwargs.get("xc", "B3LYP")
    mf = dft.KS(mol, xc=xc)

    # density fitting
    density_fit = kwargs.get("density_fit", False)
    if density_fit:
        aux_basis = kwargs.get("aux_basis", None)
        mf = mf.density_fit(auxbasis=aux_basis)
    
    # set solvation model
    solvation = kwargs.get("solvation", None)
    solvent = kwargs.get("solvent", None)
    solvent_params = kwargs.get("solvent_params", None)
    assert solvent is None or solvent_params is None, \
        "You can only specify one of --solvent or --solvent-param"
    pcm_models = {"C-PCM", "IEF-PCM", "SS(V)PE", "COSMO"}
    if solvation in pcm_models:
        eps_dict = read_pcm_eps()
        mf = mf.PCM()
        mf.with_solvent.method = solvation
        if solvent is not None:
            assert solvent.lower() in eps_dict, \
                f"Solvent {solvent} not found in predefined solvents"
            eps = eps_dict[solvent.lower()]
        elif solvent_params is not None:
            assert len(solvent_params) == 1, \
                "You must provide exactly one parameter of dielectric constant for PCM model"
            eps = solvent_params[0]
        mf.with_solvent.eps = eps
    elif solvation == "SMD":
        mf = mf.SMD()
        if solvent is not None:
            mf.with_solvent.solvent = solvent
        elif solvent_params is not None:
            assert len(solvent_params) == 8, \
                """
                You must provide exactly 8 parameters for SMD solvation model:
                [n, n25, alpha, beta, gamma, epsilon, phi, psi]
                """
            mf.with_solvent.solvent_descriptors = solvent_params
    
    # set other parameters
    mf.disp = kwargs.get("disp", None)
    mf.conv_tol = kwargs.get("scf_conv", 1e-8)
    mf.grids.grid_level = kwargs.get("grid", 9)
    mf.max_cycle = kwargs.get("scf_max_cycle", 50)
    
    return mf

