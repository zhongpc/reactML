import argparse
import time
from types import SimpleNamespace

import numpy as np
import ase.io
import yaml
import h5py
from pyscf import symm, gto
from pyscf.hessian import thermo
from tblite.interface import Calculator
from tblite.ase import TBLite
from ase import Atoms, units
from sella import Sella, IRC, Constraints
from sella.optimize.irc import IRCInnerLoopConvergenceFailure

from reactML.common.utils import dump_normal_mode


def xTB_numerical_hessian(
    atomic_numbers: np.ndarray,
    positions: np.ndarray,  # Angstrom
    init_kwargs: dict,
    set_kwargs: dict = None,
    add_kwargs: dict = None,
    h: float = 5e-3,
) -> np.ndarray:
    """
    Compute numerical Hessian using finite difference method.
    Args:
        tblite_calc: TBLite calculator instance.
        positions: Atomic positions (in Angstrom).
        h: Finite difference step size (in Angstrom).
    Returns:
        Hessian matrix (in Hartree/Bohr^2).
    """
    n_atoms = len(atomic_numbers)
    hessian = np.zeros((n_atoms, n_atoms, 3, 3))
    _positions = positions.copy() / units.Bohr  # convert to Bohr
    _h = h / units.Bohr  # convert to Bohr

    assert init_kwargs, "init_kwargs must be provided"
    for i in range(n_atoms):
        for j in range(3):
            displaced_plus = _positions.copy()
            displaced_plus[i, j] += _h
            displaced_minus = _positions.copy()
            displaced_minus[i, j] -= _h
            calc_plus = Calculator(
                numbers=atomic_numbers,
                positions=displaced_plus,
                **init_kwargs,
            )
            calc_minus = Calculator(
                numbers=atomic_numbers,
                positions=displaced_minus,
                **init_kwargs,
            )
            if set_kwargs:
                for key, value in set_kwargs.items():
                    calc_plus.set(key, value)
                    calc_minus.set(key, value)
            if add_kwargs:
                for key, value in add_kwargs.items():
                    calc_plus.add(key, value)
                    calc_minus.add(key, value)
            res_plus = calc_plus.singlepoint()
            grad_plus = res_plus["gradient"]  # in Hartree/Bohr  
            res_minus = calc_minus.singlepoint()
            grad_minus = res_minus["gradient"]  # in Hartree/Bohr
            hessian[i, :, j, :] = (grad_plus - grad_minus) / (2 * _h)
    
    return hessian


CACHED_POSITION = None
CACHED_HESSIAN = None

def hessian_function(
    atoms: Atoms,
    init_kwargs: dict,
    set_kwargs: dict = None,
    add_kwargs: dict = None,
) -> np.ndarray:
    if CACHED_POSITION is not None and np.allclose(atoms.get_positions(), CACHED_POSITION):
        return CACHED_HESSIAN
    atomic_numbers = atoms.get_atomic_numbers()
    positions = atoms.get_positions()  # in Angstrom
    hessian = xTB_numerical_hessian(
        atomic_numbers,
        positions,
        init_kwargs,
        set_kwargs,
        add_kwargs,
    )
    n_atoms = len(atomic_numbers)
    hessian = hessian.transpose(0, 2, 1, 3).reshape(3 * n_atoms, 3 * n_atoms)
    hessian *= (units.Hartree / units.Bohr**2)  # convert from Eh/Bohr^2
    return hessian


def main():
    global CACHED_POSITION, CACHED_HESSIAN
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="tblite_config.yaml",
        help="Path to the xTB config YAML file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config: dict = yaml.safe_load(f)
    
    # setup files
    inputfile: str = config.get("inputfile", "mol.xyz")
    filename = inputfile.rsplit(".", 1)[0]
    datafile: str = config.get("datafile", f"{filename}_data.h5")
    # empty datafile if save anything
    for key in config:
        if isinstance(key, str) and key.startswith("save_") and config[key]:
            h5py.File(datafile, "w").close()
            break
    
    # set symmetry tolerance (hardcoded in Angstrom)
    if "symm_geom_tol" in config:
        symm.geom.TOLERANCE = config["symm_geom_tol"] / units.Bohr

    # build method
    atoms = ase.io.read(config["inputfile"])
    if "charge" in config:
        atoms.info["charge"] = config["charge"]
    elif "charge" in atoms.info:
        config["charge"] = atoms.info["charge"]
    else:
        raise ValueError("Charge must be specified in the configuration file or in the input XYZ file.")
    if "multiplicity" in config:
        atoms.info["multiplicity"] = config["multiplicity"]
    elif "multiplicity" in atoms.info:
        config["multiplicity"] = atoms.info["multiplicity"]
    else:
        raise ValueError("Multiplicity must be specified in the configuration file or in the input XYZ file.")
    # load parameters
    method = config.get("xtb", "GFN2-xTB")
    charge = config["charge"]
    multiplicity = config["multiplicity"]
    accuracy = config.get("accuracy", 1.0)
    eTemp = config.get("eTemp", 298.15)
    max_iter = config.get("max_iter", 250)
    mixer_damping = config.get("mixer_damping", 0.4)
    electric_field = config.get("electric_field", None)
    spin_polarization = config.get("spin_polarization", None)
    alpb_solvation = config.get("alpb_solvation", None)
    cpcm_solvation = config.get("cpcm_solvation", None)
    # assert alpb_solvation and cpcm_solvation are not both set
    if alpb_solvation is not None and cpcm_solvation is not None:
        raise ValueError("Only one of alpb_solvation or cpcm_solvation can be set.")
    verbosity = config.get("verbosity", 0)
    init_kwargs = {
        "method": method,
        "charge": charge,
        "uhf": multiplicity - 1,
    }
    set_kwargs = {
        "accuracy": accuracy,
        "max-iter": max_iter,
        "mixer-damping": mixer_damping,
        "temperature": eTemp * units.kB / units.Hartree,
        "verbosity": verbosity,
    }
    add_kwargs = {}
    if electric_field is not None:
        add_kwargs["electric-field"] = electric_field
    if spin_polarization is not None:
        add_kwargs["spin-polarization"] = spin_polarization
    if alpb_solvation is not None:
        add_kwargs["alpb-solvation"] = alpb_solvation
    elif cpcm_solvation is not None:
        add_kwargs["cpcm-solvation"] = cpcm_solvation
    

    # set calculator
    calc = TBLite(
        method=method,
        charge=charge,
        multiplicity=multiplicity,
        accuracy=accuracy,
        electronic_temperature=eTemp,
        max_iterations=max_iter,
        mixer_damping=mixer_damping,
        electric_field=electric_field,
        spin_polarization=spin_polarization,
        alpb_solvation=alpb_solvation,
        cpcm_solvation=cpcm_solvation,
        verbosity=verbosity,
    )
    atoms.calc = calc

    # task 1: optimization
    run_opt = config.get("opt", False)
    if run_opt:
        # record start time
        start_time = time.time()
        # parameters for Sella
        opt_config: dict = config.get("opt_config", {})
        optts: bool = opt_config.get("ts", False)
        if optts:
            eig = opt_config.get("calc_hess", True)
            order = 1
        else:
            eig = opt_config.get("calc_hess", False)
            order = 0
        # constraints
        if "constraints" in opt_config:
            cons = Constraints(atoms)
            cons_dict: dict = opt_config["constraints"]
            # translation
            if "fix_translation" in cons_dict:
                for atom_idx in cons_dict["fix_translation"]:
                    cons.fix_translation(atom_idx)
                print(f"Applied translation constraints: {cons_dict['fix_translation']}")
            # bond
            if "fix_bond" in cons_dict:
                for bond in cons_dict["fix_bond"]:
                    cons.fix_bond((bond[0], bond[1]))
                print(f"Applied bond constraints: {cons_dict['fix_bond']}")
            # angle
            if "fix_angle" in cons_dict:
                for angle in cons_dict["fix_angle"]:
                    cons.fix_angle((angle[0], angle[1], angle[2]))
                print(f"Applied angle constraints: {cons_dict['fix_angle']}")
            # dihedral
            if "fix_dihedral" in cons_dict:
                for dihedral in cons_dict["fix_dihedral"]:
                    cons.fix_dihedral((dihedral[0], dihedral[1], dihedral[2], dihedral[3]))
                print(f"Applied dihedral constraints: {cons_dict['fix_dihedral']}")
        else:
            cons = None
        sella_opt = Sella(
            atoms=atoms,
            trajectory=opt_config.get("trajectory", f"{filename}_opt.traj"),
            order=order,  # 0 for minimum, 1 for saddle point
            internal=opt_config.get("internal", True),
            constraints=cons,
            constraints_tol=float(opt_config.get("constraints_tol", 1e-5)),
            delta0=opt_config.get("delta0", None),
            eta=float(opt_config.get("eta", 1e-4)),
            gamma=float(opt_config.get("gamma", 0.1)),
            eig=eig,
            threepoint=True,
            nsteps_per_diag=opt_config.get("nsteps_per_diag", 3),
            diag_every_n=opt_config.get("diag_every_n", None),
            hessian_function=lambda x: hessian_function(x, init_kwargs, set_kwargs, add_kwargs),
        )
        energy_criteria = float(opt_config.get("energy", 1e-6)) * units.Hartree
        fmax_criteria = float(opt_config.get("fmax", 4.5e-4)) * units.Hartree / units.Bohr
        frms_criteria = float(opt_config.get("frms", 3.0e-4)) * units.Hartree / units.Bohr
        dmax_criteria = float(opt_config.get("dmax", 1.8e-3))
        drms_criteria = float(opt_config.get("drms", 1.2e-3))
        max_steps: int = opt_config.get("max_steps", 200)
        last_pos = atoms.get_positions().copy()
        last_energy = np.inf
        for i in sella_opt.irun(fmax=0, steps=max_steps):
            delta_pos = np.linalg.norm(atoms.get_positions() - last_pos, axis=1)
            delta_energy = abs(atoms.get_potential_energy() - last_energy)
            fmax = np.max(np.abs(atoms.get_forces()))
            frms = np.sqrt(np.mean(atoms.get_forces()**2))
            dmax = np.max(delta_pos)
            drms = np.sqrt(np.mean(delta_pos**2))
            if (delta_energy < energy_criteria and
                fmax < fmax_criteria and
                frms < frms_criteria and
                dmax < dmax_criteria and
                drms < drms_criteria):
                print("Optimization converged based on given criteria.")
                break
            last_pos = atoms.get_positions().copy()
            last_energy = atoms.get_potential_energy()
        else:
            Warning("Optimization did not converge within the maximum number of steps.")
            print(f"Final Energy Change   : {delta_energy:.6e} Eh")
            print(f"Final MAX force       : {fmax * units.Bohr / units.Hartree:.6e} Eh/Bohr")
            print(f"Final RMS force       : {frms * units.Bohr / units.Hartree:.6e} Eh/Bohr")
            print(f"Final MAX displacement: {dmax:.6e} Angstrom")
            print(f"Final RMS displacement: {drms:.6e} Angstrom")
        # save final structure
        opt_outputfile = opt_config.get("outputfile", f"{filename}_opt.xyz")
        ase.io.write(opt_outputfile, atoms, columns=["symbols", "positions"])
        # record end time
        end_time = time.time()
        print(f"Optimization completed in {end_time - start_time:.2f} seconds.")

    # task 2: single point energy
    start_time = time.time()
    xtb_calc = Calculator(
        numbers=atoms.numbers,
        positions=atoms.positions / units.Bohr,
        **init_kwargs,
    )
    if set_kwargs:
        for key, value in set_kwargs.items():
            xtb_calc.set(key, value)
    if add_kwargs:
        for key, value in add_kwargs.items():
            xtb_calc.add(key, value)
    res = xtb_calc.singlepoint()
    energy = res.get("energy")  # in Hartree
    # energy = atoms.get_potential_energy() / units.Hartree
    end_time = time.time()
    print(f"Single point calculation completed in {end_time - start_time:.2f} seconds.")
    print(f"Total Energy: {energy:.6f} Eh")

    # task 3: forces (gradients)
    run_forces = config.get("forces", False)
    if run_forces:
        start_time = time.time()
        forces = -res.get("gradient")  # in Hartree/Bohr
        end_time = time.time()
        print(f"Force calculation completed in {end_time - start_time:.2f} seconds.")
        print("Forces (Eh/Bohr):")
        for i, (ele, force) in enumerate(zip(atoms.get_chemical_symbols(), forces)):
            print(f"{i+1:3d} {ele:2s} {force[0]:12.6f} {force[1]:12.6f} {force[2]:12.6f}")
        save_forces = config.get("save_forces", False)
        if save_forces:
            with h5py.File(datafile, 'a') as h5f:
                h5f.create_dataset("forces", data=forces)
                h5f.create_dataset("forces_unit", data="Eh/Bohr")

    # task 4: vibrational frequency analysis
    run_freq = config.get("freq", False)
    freq_config = config.get("freq_config", {})
    if run_freq:
        # calculate Hessian matrix
        start_time = time.time()
        hessian = xTB_numerical_hessian(
            atoms.get_atomic_numbers(),
            atoms.get_positions(),
            init_kwargs,
            set_kwargs,
            add_kwargs,
        )
        end_time = time.time()
        print(f"Hessian calculation completed in {end_time - start_time:.2f} seconds.")

        CACHED_POSITION = atoms.get_positions().copy()
        _hessian = hessian.transpose(0, 2, 1, 3).reshape(3 * len(atoms), 3 * len(atoms))
        CACHED_HESSIAN = _hessian * (units.Hartree / units.Bohr**2)  # Convert from Hartree/Bohr^2
        # (optional) save Hessian matrix (a.u.)
        save_hess: bool = config.get("save_hess", False)
        if save_hess:
            with h5py.File(datafile, 'a') as h5f:
                h5f.create_dataset("hessian", data=hessian)
                h5f.create_dataset("hessian_unit", data="Eh/Bohr^2")
        
        # vibrational analysis
        start_time = time.time()
        mol = gto.M(
            atom=[(ele, coord) for ele, coord in zip(atoms.get_chemical_symbols(), atoms.get_positions())],
            charge=charge,
            spin=multiplicity - 1,
        )
        freq_info = thermo.harmonic_analysis(mol, hessian, imaginary_freq=False)
        # imaginary frequencies
        freq_au = freq_info["freq_au"]
        num_imag = np.sum(freq_au < 0)
        if num_imag > 0:
            print(f"Note: {num_imag} imaginary frequencies detected!")
        dummy_mf = SimpleNamespace(mol=mol, e_tot=energy)
        temp = freq_config.get("temp", 298.15)
        press = freq_config.get("press", 101325)
        thermo_info = thermo.thermo(dummy_mf, freq_au, temp, press)
        end_time = time.time()
        print(f"Vibrational frequency analysis completed in {end_time - start_time:.2f} seconds.")
        # log thermo info
        dump_normal_mode(mol, freq_info)
        thermo.dump_thermo(mol, thermo_info)
        # save frequencies and normal modes
        save_freq: bool = config.get("save_freq", False)
        if save_freq:
            with h5py.File(datafile, 'a') as h5f:
                h5f.create_dataset("freq_wavenumber", data=freq_info["freq_wavenumber"])
                h5f.create_dataset("freq_wavenumber_unit", data="cm^-1")
                h5f.create_dataset("norm_mode", data=freq_info["norm_mode"])
        # save thermo data
        save_thermo: bool = config.get("save_thermo", False)
        if save_thermo:
            pyscf_names = ["temperature", "pressure", "E0", "ZPE", "E_tot", "H_tot", "S_tot", "G_tot"]
            reactml_names = ["T", "P", "E0", "ZPE", "U", "H", "S", "G"]
            with h5py.File(datafile, 'a') as h5f:
                for pyscf_name, reactml_name in zip(pyscf_names, reactml_names):
                    h5f.create_dataset(reactml_name, data=thermo_info[pyscf_name][0])
                    h5f.create_dataset(f"{reactml_name}_unit", data=thermo_info[pyscf_name][1])
    
    # task 5: IRC
    run_irc = config.get("irc", False)
    irc_trajectory: str = config.get("irc_trajectory", f"{filename}_irc.traj")
    if run_irc:
        start_time = time.time()
        irc_config: dict = config.get("irc_config", {})
        sella_irc = IRC(
            atoms=atoms,
            trajectory=irc_trajectory,
            ninner_iter=irc_config.get("ninner_iter", 10),
            dx=float(irc_config.get("dx", 0.1)),
            eta=float(irc_config.get("eta", 1e-4)),
            peskwargs={"threepoint": True},
            keep_going=irc_config.get("keep_going", False),
            diag_every_n=irc_config.get("diag_every_n", None),
            hessian_function=lambda x: hessian_function(x, init_kwargs, set_kwargs, add_kwargs),
        )
        fmax: float = float(irc_config.get("fmax", 4.5e-4)) * units.Hartree / units.Bohr
        irc_steps: int = irc_config.get("irc_steps", 10)
        direction: str = irc_config.get("direction", "both")
        assert direction in ["forward", "reverse", "both"], "Invalid IRC direction. Choose from 'forward', 'reverse', or 'both'."

        # reverse direction
        # record the initial position
        pos_init = atoms.get_positions().copy()
        if direction in ["reverse", "both"]:
            print("Starting reverse IRC")
            try:
                irc_converged = sella_irc.run(fmax=fmax, steps=irc_steps, direction="reverse")
            except IRCInnerLoopConvergenceFailure as e:
                Warning("IRC inner loop failed to converge: " + str(e))
                irc_converged = False
            finally:
                print(f"IRC completed. Converged: {irc_converged}")
                ase.io.write(f"{filename}_irc_reverse.xyz", sella_irc.atoms, format="xyz")
            atoms_traj = ase.io.Trajectory(irc_trajectory, mode="r")
            reverse_steps = sella_irc.nsteps
        else:
            reverse_steps = 0
        
        # forward direction
        if direction in ["forward", "both"]:
            print("Starting forward IRC")
            # reset to initial position
            sella_irc.v0ts = None
            atoms.set_positions(pos_init)
            # reset trajectory
            try:
                irc_converged = sella_irc.run(fmax=fmax, steps=irc_steps, direction="forward")
            except IRCInnerLoopConvergenceFailure as e:
                Warning("IRC inner loop failed to converge: " + str(e))
                irc_converged = False
            finally:
                print(f"IRC completed. Converged: {irc_converged}")
                ase.io.write(f"{filename}_irc_forward.xyz", sella_irc.atoms, format="xyz")
            atoms_traj = ase.io.Trajectory(irc_trajectory, mode="r")
            print(len(atoms_traj), "frames saved in", irc_trajectory)
            print(sella_irc.nsteps, "IRC steps taken.")
            forward_steps = sella_irc.nsteps
        else:
            forward_steps = 0
        end_time = time.time()
        print(f"IRC calculation completed in {end_time - start_time:.2f} seconds.")

        save_traj_xyz: bool = irc_config.get("save_traj_xyz", False)
        if save_traj_xyz:
            atoms_traj = ase.io.Trajectory(irc_trajectory, mode="r")
            atoms_traj = list(atoms_traj)
            atoms_ts = atoms_traj[0]
            atoms_ts.info["note"] = "TS"
            # reverse the reverse IRC steps to maintain chronological order
            atoms_list = atoms_traj[1:reverse_steps+1][::-1]
            # append the TS structure
            atoms_list.append(atoms_ts)
            # append the forward IRC steps
            atoms_list.extend(atoms_traj[reverse_steps+1:reverse_steps+1+forward_steps])
            ase.io.write(f"{filename}_irc.xyz", atoms_list)
            print(f"Full IRC path saved to {filename}_irc.xyz")


if __name__ == "__main__":
    main()