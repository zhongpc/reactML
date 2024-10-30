#!/usr/bin/env python
#
# Author: Garnet Chan <gkc1000@gmail.com>
# adapted by Stefan Heinen <heini.phys.chem@gmail.com>:
# - added forces
# - changed the ase_atoms_to_pyscf function
# - added an mp2 wrapper

'''
ASE package interface
'''

import numpy as np
from ase.calculators.calculator import Calculator
import pyscf

def get_mp2_energy():
	test = 1
	

def xyz_to_string(atoms):
    """Convert ASE Atoms object to a formatted string representation."""
    lines = []
    for atom in atoms:
        symbol = atom.symbol
        x, y, z = atom.position
        lines.append(f"   {symbol:<2s}        {x:>10.5f}   {y:>10.5f}   {z:>10.5f}")

    return "\n" + "\n".join(lines) 

def ase_atoms_to_pyscf(ase_atoms):
		'''Convert ASE atoms to PySCF atom.

		Note: ASE atoms always use A.
		'''
#return [[ase_atoms.get_chemical_symbols(), ase_atoms.get_positions()] for i, atom in enumerate(ase_atoms)]
		return [ [ase_atoms.get_chemical_symbols()[i], ase_atoms.get_positions()[i]] for i in range(len(ase_atoms.get_positions()))]

atoms_from_ase = ase_atoms_to_pyscf

class PySCF_calculator(Calculator):
		implemented_properties = ['energy', 'forces']

		def __init__(self, restart=None, ignore_bad_restart_file=False,
								 label='PySCF', atoms=None, scratch=None, **kwargs):
				"""Construct PySCF-calculator object.

				Parameters
				==========
				label: str
						Prefix to use for filenames (label.in, label.txt, ...).
						Default is 'PySCF'.

				mfclass: PySCF mean-field class
				molcell: PySCF :Mole: or :Cell:
				"""
				Calculator.__init__(self, restart=None, ignore_bad_restart_file=False,
														label='PySCF', atoms=None, scratch=None, **kwargs)

				self.mf=None
				self.initialize(**kwargs)

		def initialize(self, 
				#  molcell, 
				 mf_class,
				 bas = 'def2-svpd',
				 max_memory = 32000):
				
                # Compute Energy
                # e_dft = mf_GPU.kernel()
                # print(f"total energy = {e_dft}") # -76.26736519501688

                # # Compute Gradient
                # g = mf_GPU.nuc_grad_method()
                # g.max_memory = 20000
                # g.auxbasis_response = True
                # g_dft = g.kernel()

#				if not molcell.unit.startswith(('A','a')):
#						raise RuntimeError("PySCF unit must be A to work with ASE")

				# self.molcell=molcell
				self.mf_class=mf_class
				self.bas = bas
				self.max_memory = max_memory
				
		def set(self, **kwargs):
				changed_parameters = Calculator.set(self, **kwargs)
				if changed_parameters:
						self.reset()

		def calculate(self, atoms=None, properties=['energy', 'forces'],
									system_changes=['positions', 'numbers', 'cell',
																	'pbc', 'charges','magmoms']):

				Calculator.calculate(self, atoms)
				# pyscf_atom = ase_atoms_to_pyscf(atoms)
				atom_string = xyz_to_string(atoms)
				mol = pyscf.M(atom=atom_string, basis= self.bas, max_memory= self.max_memory)
				self.mf = self.mf_class(mol)

				# calc_molcell = self.molcell.copy()
				# calc_molcell.atom = ase_atoms_to_pyscf(atoms)
				# calc_molcell.build(None,None)
				# self.mf = self.mf_class(calc_molcell)
				e_dft = self.mf.kernel()
				g = self.mf.nuc_grad_method()
				g.max_memory = 20000
				g.auxbasis_response = True
				g_dft = g.kernel()


				self.results['energy']= e_dft # self.mf.scf(verbose=0)
				self.results['forces']=-1 * g_dft # convert forces to gradient (*-1) !!!!! for the NEB run
				# self.results['mf']=self.mf