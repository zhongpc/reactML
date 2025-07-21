from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes

from pyscf.gto import Mole
from pyscf.grad.rhf import symmetrize
from pyscf.lib import GradScanner


class PySCFCalculator(Calculator):
    """
    PySCF calculator for ASE.
    This calculator uses PySCF to compute the energy and forces of a system.
    It can be used with various mean field methods provided by PySCF.
    """
    implemented_properties = ["energy", "forces"]
    default_parameters = {
        "method": None,  # PySCF mean field method class
    }
    def __init__(self, **kwargs):
        self.method = None
        self.g_scanner: GradScanner = None
        Calculator.__init__(self, **kwargs)

    def set(self, **kwargs):
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
        if "method" in changed_parameters:
            self.method = kwargs["method"]
            self.g_scanner = self.method.nuc_grad_method().as_scanner()

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
        coords = atoms.get_positions()

        if mol.symmetry:
            coords = symmetrize(mol, coords)
        
        mol.set_geom_(coords, unit="Angstrom")
        
        energy, gradients = self.g_scanner(mol)

        # store the energy and forces
        self.results["energy"] = energy * units.Hartree  # convert energy from Hartree to eV
        self.results["forces"] = -gradients * units.Hartree / units.Bohr  # convert forces from Hartree/Bohr to eV/Angstrom

