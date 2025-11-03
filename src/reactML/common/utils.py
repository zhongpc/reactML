import os
from types import MethodType
from typing import Dict

from pyscf import gto, lib, dft


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
    pcm_eps_txt = os.path.join(os.path.dirname(__file__), "pcm_eps.txt")
    with open(pcm_eps_txt, "r") as f:
        lines = f.readlines()
    eps_dict = {}
    for line in lines:
        solvent, eps = line.split(": ")
        eps_dict[solvent.strip().lower()] = float(eps.strip())
    return eps_dict


def build_method(config: dict):
    """
    Create a PySCF mean field object from a configuration dictionary.

    Args:
        config (dict): Configuration dictionary containing parameters for the mean field object.

    Returns:
        mf (pyscf.dft.KS): The mean field object.
    """
    xc = config.get("xc", "B3LYP")
    basis = config.get("basis", "def2-SVP")
    ecp = config.get("ecp", None)
    nlc = config.get("nlc", "")
    disp = config.get("disp", None)
    grids = config.get("grids", {"atom_grid": (99, 590)})
    nlcgrids = config.get("nlcgrids", {"atom_grid": (50, 194)})
    verbose = config.get("verbose", 4)
    scf_conv_tol = config.get("scf_conv_tol", 1e-8)
    direct_scf_tol = config.get("direct_scf_tol", 1e-8)
    scf_max_cycle = config.get("scf_max_cycle", 50)
    with_df = config.get("with_df", True)
    auxbasis = config.get("auxbasis", "def2-universal-jkfit")
    with_gpu = config.get("with_gpu", True)
    
    with_solvent = config.get("with_solvent", False)
    solvent = config.get("solvent", {"method": "ief-pcm", "eps": 78.3553, "solvent": "water"})
    
    max_memory = config.get("max_memory", None)
    if max_memory is not None:
        max_memory *= 1024  # convert GB to MB
    threads = config.get("threads", os.environ.get("OMP_NUM_THREADS", os.cpu_count()))
    lib.num_threads(threads)

    atom = config.get("inputfile", "mol.xyz")
    charge = config.get("charge", 0)
    spin = config.get("spin", None)

    # build molecule
    mol = gto.M(
        atom=atom,
        basis=basis,
        ecp=ecp,
        max_memory=max_memory,
        verbose=verbose,
        charge=charge,
        spin=spin,
    )
    mol.build()

    # build Kohn-Sham object
    mf = dft.KS(mol, xc=xc)
    mf.nlc = nlc
    mf.disp = disp
    # set grids
    if "atom_grid" in grids:
        mf.grids.atom_grid = grids["atom_grid"]
    if "level" in grids:
        mf.grids.level = grids["level"]
    if mf._numint.libxc.is_nlc(mf.xc) or nlc is not None:
        if "atom_grid" in nlcgrids:
            mf.nlcgrids.atom_grid = nlcgrids["atom_grid"]
        if "level" in nlcgrids:
            mf.nlcgrids.level = nlcgrids["level"]
    # set density fitting
    if with_df:
        mf = mf.density_fit(auxbasis=auxbasis)
    
    # move to GPU if available
    if with_gpu:
        try:
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
            mf = mf.to_gpu()
        except ImportError:
            print("GPU support is not available. Proceeding with CPU.")
    
    # solvation model
    if with_solvent:
        solvent = solvent
        if solvent["method"].upper() in {"C-PCM", "IEF-PCM", "SS(V)PE", "COSMO"}:
            mf = mf.PCM()
            mf.with_solvent.lebedev_order = 29
            mf.with_solvent.method = solvent["method"]
            if "eps" in solvent:
                mf.with_solvent.eps = solvent["eps"]
            elif "solvent" in solvent:
                eps_dict = read_pcm_eps()
                assert solvent["solvent"].lower() in eps_dict, \
                    f"Solvent {solvent['solvent']} not found in predefined solvents"
                mf.with_solvent.eps = eps_dict[solvent["solvent"].lower()]
            else:
                raise ValueError("You must provide either 'eps' or 'solvent' for PCM model.")
        elif solvent["method"].upper() == "SMD":
            mf = mf.SMD()
            mf.with_solvent.lebedev_order = 29
            mf.with_solvent.method = "SMD"
            if "solvent_descriptors" in solvent:
                mf.with_solvent.solvent_descriptors = solvent["solvent_descriptors"]
            elif "solvent" in solvent:
                mf.with_solvent.solvent = solvent["solvent"]
            else:
                raise ValueError("You must provide either 'solvent_descriptors' or 'solvent' for SMD model.")
        else:
            raise ValueError(f"Solvation method {solvent['method']} not recognized.")
        
    mf.direct_scf_tol = float(direct_scf_tol)
    mf.chkfile = None
    mf.conv_tol = float(scf_conv_tol)
    mf.max_cycle = scf_max_cycle

    return mf


def build_3c_method(config: dict):
    """
    Special cases for 3c methods, e.g., B97-3c
    """
    xc = config.get("xc", "B97-3c")
    if not xc.endswith("3c"):
        raise ValueError("The xc functional must be a 3c method, e.g., B97-3c.")
    from gpu4pyscf.drivers.dft_3c_driver import parse_3c, gen_disp_fun
    
    # modify config dictionary
    pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp = parse_3c(xc.lower())
    config["xc"] = pyscf_xc
    config["nlc"] = nlc
    config["basis"] = basis
    config["ecp"] = ecp

    # build method
    mf = build_method(config)

    # attach 3c specific functions
    mf.get_dispersion = MethodType(gen_disp_fun(xc_disp, xc_gcp), mf)
    mf.do_disp = lambda: True

    return mf


def get_gradient_method(mf, xc_3c=None):
    """
    Get the gradient method from a mean field object.
    Args:
        mf (pyscf.dft.KS): The mean field object.
    Returns:
        grad (pyscf.grad.KS): The gradient method.
    """
    # 3c methods
    if xc_3c is not None:
        if not xc_3c.endswith("3c"):
            raise ValueError("The xc functional must be a 3c method, e.g., B97-3c.")
        from gpu4pyscf.drivers.dft_3c_driver import parse_3c, gen_disp_grad_fun
        _, _, _, _, (xc_disp, disp), xc_gcp = parse_3c(xc_3c.lower())
        g = mf.nuc_grad_method()
        g.get_dispersion = MethodType(gen_disp_grad_fun(xc_disp, xc_gcp), g)
        return g
    
    return mf.nuc_grad_method()


def get_Hessian_method(mf, xc_3c=None):
    """
    Get the Hessian method from a mean field object.
    Args:
        mf (pyscf.dft.KS): The mean field object.
    Returns:
        hess (pyscf.hessian.KS): The Hessian method.
    """
    # 3c methods
    if xc_3c is not None:
        if not xc_3c.endswith("3c"):
            raise ValueError("The xc functional must be a 3c method, e.g., B97-3c.")
        from gpu4pyscf.drivers.dft_3c_driver import parse_3c, gen_disp_hess_fun
        _, _, _, _, (xc_disp, disp), xc_gcp = parse_3c(xc_3c.lower())
        h = mf.Hessian()
        h.get_dispersion = MethodType(gen_disp_hess_fun(xc_disp, xc_gcp), h)
        h.auxbasis_response = 2
        return h
    
    h = mf.Hessian()
    h.auxbasis_response = 2
    return h