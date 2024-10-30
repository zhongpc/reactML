import numpy as np
from ase.calculators.calculator import Calculator
import pyscf


def ase_to_string(atoms):
	"""Convert ASE Atoms object to a formatted string representation."""
	lines = []
	for atom in atoms:
		symbol = atom.symbol
		x, y, z = atom.position
		lines.append(f"   {symbol:<2s}        {x:>10.5f}   {y:>10.5f}   {z:>10.5f}")

	return "\n" + "\n".join(lines) 


class PySCF_calculator(Calculator):
	implemented_properties = ['energy', 'forces']

	def __init__(self, restart=None, ignore_bad_restart_file=False, label='PySCF', atoms=None, scratch=None, **kwargs):
		"""
			Construct PySCF-calculator object.
		"""
		Calculator.__init__(self, restart=None, ignore_bad_restart_file=False, label='PySCF', atoms=None, scratch=None, **kwargs)

		self.mf = None
		self.initialize(**kwargs)

	def initialize(self, 
				   #  molcell, 
				   mf_class,
				   bas='def2-svpd',
				   max_memory=32000):
		"""	
			Initialize the PySCF calculator.
			Args:
				molcell: PySCF Mole object
				mf_class: PySCF mean feild class
				bas: basis set
				max_memory: maximum memory
		"""
		self.mf_class = mf_class
		self.bas = bas
		self.max_memory = max_memory

	def set(self, **kwargs):
		changed_parameters = Calculator.set(self, **kwargs)
		if changed_parameters:
			self.reset()

	def calculate(self, atoms=None, properties=['energy', 'forces'], 
			   	  system_changes=['positions', 'numbers', 'cell', 'pbc', 'charges', 'magmoms']):

		Calculator.calculate(self, atoms)
		atom_string = ase_to_string(atoms)
		mol = pyscf.M(atom=atom_string, basis=self.bas, max_memory=self.max_memory)
		self.mf = self.mf_class(mol)

		# calc_molcell = self.molcell.copy()
		# calc_molcell.atom = ase_atoms_to_pyscf(atoms)
		# calc_molcell.build(None,None)
		# self.mf = self.mf_class(calc_molcell)
		
		# compute the energy
		e_dft = self.mf.kernel() 

		# compute the gradients
		g = self.mf.nuc_grad_method()
		g.max_memory = 20000
		g.auxbasis_response = True
		g_dft = g.kernel()

		# store the energy and forces
		self.results['energy'] = e_dft  # self.mf.scf(verbose=0)
		self.results['forces'] = -1 * g_dft  # convert forces to gradient (*-1) 
		# self.results['mf']=self.mf
