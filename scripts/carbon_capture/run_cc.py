import argparse
from typing import Tuple

import numpy as np
from pyscf import gto
from pyscf import cc, scf
from scipy.optimize import curve_fit


CBS_MAP = {
    "cc": {
        "dz": "cc-pVDZ", "tz": "cc-pVTZ", "qz": "cc-pVQZ", "alpha": 5.46,
    },
    "aug-cc": {
        "dz": "aug-cc-pVDZ", "tz": "aug-cc-pVTZ", "qz": "aug-cc-pVQZ", "alpha": 5.46,
    },
    "def2": {
        "dz": "def2-SVP", "tz": "def2-TZVP", "qz": "def2-QZVP", "alpha": 7.88,
    },
    "ano": {
        "dz": "ano-pVDZ", "tz": "ano-pVTZ", "qz": "ano-pVQZ", "alpha": 4.48,
    },
    "saug-ano": {
        "dz": "saug-ano-pVDZ", "tz": "saug-ano-pVTZ", "qz": "saug-ano-pVQZ", "alpha": 4.18,
    },
    "aug-ano": {
        "dz": "aug-ano-pVDZ", "tz": "aug-ano-pVTZ", "qz": "aug-ano-pVQZ", "alpha": 5.12,
    },
}


def run_hf(mol, **kwargs) -> scf.hf.SCF:
    """
    Run Hartree-Fock calculation on the given molecule.
    Args:
        mol (gto.Mole): The molecule object.
        kwargs: Additional parameters for the HF calculation.
    Returns:
        mf (pyscf.scf.hf.SCF): The Hartree-Fock object.
    """
    # build mol
    max_memory = kwargs.get("max_memory", None)
    if max_memory is not None:
        max_memory *= 1024  # convert GB to MB
    basis = kwargs.get("basis", "cc-pVDZ")
    charge = kwargs.get("charge", 0)
    spin = kwargs.get("spin", 0)
    mol.build(basis=basis, charge=charge, spin=spin, max_memory=max_memory)

    # set Hartree-Fock
    mf = scf.HF(mol)
    density_fit = kwargs.get("density_fit", False)
    if density_fit:
        aux_basis = kwargs.get("aux_basis", None)
        mf = mf.density_fit(auxbasis=aux_basis)
    mf.conv_tol = kwargs.get("scf_conv", 1e-6)
    mf.max_cycle = kwargs.get("scf_max_cycle", 200)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF calculation did not converge")
    
    return mf


def run_cc(mol, **kwargs) -> Tuple[cc.ccsd.CCSDBase, float]:
    """
    Run Coupled Cluster calculation on the given molecule.
    Args:
        mol (gto.Mole): The molecule object.
        kwargs: Additional parameters for the CC calculation.
    Returns:
        mycc (pyscf.cc.CCSDBase): The CCSD object.
        et (float): The CCSD(T) energy correction.
    """
    # run Hartree-Fock calculation
    mf = run_hf(mol, **kwargs)

    # set coupled cluster
    cc_method = kwargs.get("cc", "CCSD")
    mycc = cc.CCSD(mf)
    density_fit = kwargs.get("density_fit", False)
    if density_fit:
        aux_basis = kwargs.get("aux_basis", None)
        mycc = mycc.density_fit(auxbasis=aux_basis)
    mycc.conv_tol = kwargs.get("cc_conv", 1e-7)
    mycc.max_cycle = kwargs.get("cc_max_cycle", 50)
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CC calculation did not converge")
    if cc_method == "CCSD(T)":
        et = mycc.ccsd_t()
    else:
        et = 0.0
    
    # return results
    return mycc, et


def hf_cbs_model(l, e_cbs, A, alpha):
    """
    Hartree-Fock CBS model.
    Args:
        l (int): Highest angular momentum of the basis set.
        e_cbs (float): The CBS energy.
        A (float): The A parameter for the model.
        alpha (float): The alpha parameter for the model.
    Returns:
        float: The extrapolated energy.
    """
    return e_cbs + A * np.exp(-alpha * np.sqrt(l))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "xyzfile", type=str,
        help="Input xyz file",
    )
    parser.add_argument(
        "--cc", type=str, default="CCSD", choices=["CCSD", "CCSD(T)"],
        help="Type of Coupled Cluster method to use (default CCSD)"
    )
    parser.add_argument(
        "--basis", "-b", type=str, default="cc-pVDZ",
        help="Name of Basis Set",
    )
    parser.add_argument(
        "--cbs", action="store_true",
        help="Use CBS extrapolation"
    )
    parser.add_argument(
        "--charge", "-c", type=int, default=0,
        help="Total charge"
    )
    parser.add_argument(
        "--spin", "-s", type=int, default=0,
        help="Total spin (2S not 2S+1)",
    )
    parser.add_argument(
        "--density-fit", action="store_true",
        help="Use density fitting for the calculation",
    )
    parser.add_argument(
        "--aux-basis", type=str, default=None,
        help="Auxiliary basis set for density fitting (default None)",
    )
    parser.add_argument(
        "--scf-conv", type=float, default=1e-6,
        help="SCF convergence threshold (default 1e-6 a.u.)",
    )
    parser.add_argument(
        "--scf-max-cycle", type=int, default=200,
        help="Maximum number of SCF cycles (default 200)",
    )
    parser.add_argument(
        "--cc-conv", type=float, default=1e-7,
        help="CC convergence threshold (default 1e-7 a.u.)",
    )
    parser.add_argument(
        "--cc-max-cycle", type=int, default=50,
        help="Maximum number of CC cycles (default 50)",
    )
    parser.add_argument(
        "--extrapolate-hf", action="store_true",
        help="Whether to extrapolate HF energies for CBS calculations",
    )
    args = parser.parse_args()

    # read the xyz file
    mol = gto.Mole(charge=args.charge, spin=args.spin)  # mol.build() will be called in mol.fromfile
    mol.fromfile(filename=args.xyzfile)                 # we must set charge and spin before reading the file

    params = vars(args)
    # if basis is not CBS, then we just run a single CC calculation
    if not args.cbs:
        mycc, et = run_cc(mol, **params)
        print(f"HF   Energy              : {mycc.e_hf:10.6f} a.u.")
        print(f"Corr Energy              : {mycc.e_corr:10.6f} a.u.")
        print(f"CCSD Energy              : {mycc.e_tot:10.6f} a.u.")
        if args.cc == "CCSD(T)":
            print(f"CCSD(T) Perturbative     : {et:10.6f} a.u.")
            print(f"CCSD(T) Energy           : {mycc.e_tot + et:10.6f} a.u.")
        return
    
    # if CBS, we run a series of CC calculations with different basis sets
    basis_series = params.pop("basis")
    if args.extrapolate_hf:
        mf_dz = run_hf(mol, basis=CBS_MAP[basis_series]["dz"], **params)
    mycc_tz, et_tz = run_cc(mol, basis=CBS_MAP[basis_series]["tz"], **params)
    mycc_qz, et_qz = run_cc(mol, basis=CBS_MAP[basis_series]["qz"], **params)

    if args.extrapolate_hf:  # extrapolate the HF energy
        e_hf = np.array([mf_dz.e_tot, mycc_tz.e_hf, mycc_qz.e_hf])
        l = np.array([2.0, 3.0, 4.0])
        # initial guess for the parameters
        p0 = np.array([mycc_qz.e_hf, 1.0, CBS_MAP[basis_series]["alpha"]])
        popt, _ = curve_fit(hf_cbs_model, l, e_hf, p0=p0)
        e_hf_cbs, A_fit, alpha_fit = popt
        print(f"HF   Energy DZ           : {mf_dz.e_tot:10.6f} a.u.")
        print(f"HF   Energy TZ           : {mycc_tz.e_hf:10.6f} a.u.")
        print(f"HF   Energy QZ           : {mycc_qz.e_hf:10.6f} a.u.")
        print(f"HF   Energy CBS          : {e_hf_cbs:10.6f} a.u.       (A={A_fit:.4f}, alpha={alpha_fit:.4f})")
    else:  # use QZ HF energy as the CBS value
        print(f"HF   Energy QZ           : {mycc_qz.e_hf:10.6f} a.u.")
        e_hf_cbs = mycc_qz.e_hf

    e_corr_tz = mycc_tz.e_corr + et_tz
    e_corr_qz = mycc_qz.e_corr + et_qz
    # e_corr_cbs = (4^3 * e_corr_qz - 3^3 * e_corr_tz) / (4^3 - 3^3)
    e_corr_cbs = (64.0 * e_corr_qz - 27.0 * e_corr_tz) / 37.0
    print(f"Corr Energy TZ           : {e_corr_tz:10.6f} a.u.")
    print(f"Corr Energy QZ           : {e_corr_qz:10.6f} a.u.")
    print(f"Corr Energy CBS          : {e_corr_cbs:10.6f} a.u.")
    print(f"{args.cc} Energy           : {e_hf_cbs + e_corr_cbs:10.6f} a.u.")


if __name__ == "__main__":
    main()