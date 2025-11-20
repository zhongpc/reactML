import numpy as np
from pyscf.gto import charge, Mole
from pyscf.lib import GradScanner
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes

from reactML.common.utils import get_gradient_method

class PySCFCalculator(Calculator):
    """
    PySCF calculator for ASE.
    This calculator uses PySCF to compute the energy and forces of a system.
    It can be used with various mean field methods provided by PySCF.
    """
    implemented_properties = ["energy", "forces"]
    default_parameters = {}
    def __init__(self, method, xc_3c=None, soscf=False, **kwargs):
        self.method = method
        self.g_scanner: GradScanner = get_gradient_method(self.method, xc_3c).as_scanner()
        self.soscf = soscf
        Calculator.__init__(self, **kwargs)

    def set(self, **kwargs):
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def calculate(
        self,
        atoms: Atoms = None,
        properties=None, 
        system_changes=all_changes,
    ):
        if properties is None:
            properties = self.implemented_properties
        
        Calculator.calculate(self, atoms, properties, system_changes)
        
        mol: Mole = self.method.mol
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        Z = np.array([charge(x) for x in mol.elements])
        if all(Z == atomic_numbers):
            _atoms = positions
        else:
            _atoms = list(zip(atomic_numbers, positions))
        
        mol.set_geom_(_atoms, unit="Angstrom")
        
        energy, gradients = self.g_scanner(mol)
        if not self.g_scanner.converged and self.soscf:
            # try SOSCF if not converged
            mo_init = self.g_scanner.base.mo_coeff
            mocc_init = self.g_scanner.base.mo_occ
            newton_method = self.method.newton()
            newton_method.reset(mol)
            newton_method.kernel(mo_init, mocc_init)
            self.g_scanner.base.mo_coeff = newton_method.mo_coeff
            self.g_scanner.base.mo_occ = newton_method.mo_occ
            energy, gradients = self.g_scanner(mol)
            if self.g_scanner.converged:
                print("SOSCF converged.")
        
        # store the energy and forces
        self.results["energy"] = energy * units.Hartree
        self.results["forces"] = -gradients * (units.Hartree / units.Bohr)
