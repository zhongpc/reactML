import os.path as osp
from typing import Dict, Union

import numpy as np
from pyscf import gto, lib
try:
    from gpu4pyscf import dft
except:
    from pyscf import dft


def write_xyz(mol: gto.Mole, filename: str) -> None:
    elements = mol.elements
    coords = mol.atom_coords(unit="Angstrom")
    with open(filename, "w") as f:
        f.write(f"{len(elements)}\n")
        # Convert spin (2S) to multiplicity (2S+1) format, as required by quantum chemistry conventions.
        f.write(f"{mol.charge} {mol.spin + 1}\n")
        for ele, coord in zip(elements, coords):
            f.write(f"{ele: <2}    {coord[0]:10.6f}    {coord[1]:10.6f}    {coord[2]:10.6f}\n")


def read_pcm_eps() -> Dict[str, float]:
    # from https://gaussian.com/scrf/
    pcm_eps_txt = osp.join(osp.dirname(__file__), "pcm_eps.txt")
    with open(pcm_eps_txt, "r") as f:
        lines = f.readlines()
    eps_dict = {}
    for line in lines:
        solvent, eps = line.split(": ")
        eps_dict[solvent.strip().lower()] = float(eps.strip())
    return eps_dict


def build_dft(mol: gto.Mole, **kwargs) -> Union[dft.rks.RKS, dft.uks.UKS]:
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
    disp = kwargs.get("disp", None)
    if disp is not None:
        mf.disp = disp.lower()
    mf.conv_tol = kwargs.get("scf_conv", 1e-8)
    mf.grids.level = kwargs.get("grid", 9)
    mf.max_cycle = kwargs.get("scf_max_cycle", 50)
    
    return mf


def dump_normal_mode(mol: gto.Mole, results: Dict[str, np.ndarray]) -> None:
    """
    The function in PySCF does not dump imagnary frequencies.
    We made a custom function to dump all frequencies and normal modes.
    And decoupled from PySCF mole for reusability.

    Args:
        mol (gto.Mole): The molecule object.
        results (Dict[str, np.ndarray]): A dictionary containing frequencies and normal modes.
    """

    dump = mol.stdout.write
    freq_wn = results['freq_wavenumber']
    if np.iscomplexobj(freq_wn):
        freq_wn = freq_wn.real - abs(freq_wn.imag)
    nfreq = freq_wn.size

    r_mass = results['reduced_mass']
    force = results['force_const_dyne']
    vib_t = results['vib_temperature']
    mode = results['norm_mode']
    symbols = [mol.atom_symbol(i) for i in range(mol.natm)]

    def inline(q, col0, col1):
        return ''.join('%20.4f' % q[i] for i in range(col0, col1))
    def mode_inline(row, col0, col1):
        return '  '.join('%6.2f%6.2f%6.2f' % (mode[i,row,0], mode[i,row,1], mode[i,row,2])
                         for i in range(col0, col1))

    for col0, col1 in lib.prange(0, nfreq, 3):
        dump('Mode              %s\n' % ''.join('%20d'%i for i in range(col0,col1)))
        dump('Irrep\n')
        dump('Freq [cm^-1]          %s\n' % inline(freq_wn, col0, col1))
        dump('Reduced mass [au]     %s\n' % inline(r_mass, col0, col1))
        dump('Force const [Dyne/A]  %s\n' % inline(force, col0, col1))
        dump('Char temp [K]         %s\n' % inline(vib_t, col0, col1))
        #dump('IR\n')
        #dump('Raman\n')
        dump('Normal mode            %s\n' % ('       x     y     z'*(col1-col0)))
        for j, at in enumerate(symbols):
            dump('    %4d%4s               %s\n' % (j, at, mode_inline(j, col0, col1)))

