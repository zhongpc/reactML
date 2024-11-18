# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


"""
A module to calculate free energies using the Quasi-Rigid Rotor Harmonic
 Oscillator approximation. Modified from a script by Steven Wheeler.
See: Grimme, S. Chem. Eur. J. 2012, 18, 9955
"""

import numpy as np

# Define useful constants
kb = 1.380662E-23  # J/K
c = 29979245800  # cm/s
h = 6.62606957E-34  # J.s
R = 1.987204  # kcal/mol

# Define useful conversion factors
cm2kcal = 0.0028591459
cm2hartree = 4.5563352812122E-06
kcal2hartree = 0.0015936
amu2kg = 1.66053886E-27
bohr2angs = 0.52917721092
j2hartree = 2.294E+17


class ShiftedRRHO:
    """
    Class to calculate thermochemistry using Cramer & Truhlar's RRHO approximation
    """

    def __init__(self, output, sigma_r=1, temp=298.15, press=101317, conc=1, v0=100, linear=False):
        """
        :param output: Requires dictionary of necessary inputs:
                        {"mol": Molecule, "mult": spin multiplicity (int),
                        "frequencies": list of vibrational frequencies [a.u.],
                        elec_energy": electronic energy [a.u.]}
        :param sigma_r: Rotational symmetry number
        :param temp: Temperature [K]
        :param press: Pressure [Pa]
        :param conc: Solvent concentration [M]
        :param v0: Cutoff frequency for Quasi-RRHO method [cm^1]
        """

        # TO-DO: calculate sigma_r with PointGroupAnalyzer
        # and/or edit Gaussian and QChem io to parse for sigma_r

        self.sigma_r = sigma_r
        self.temp = temp
        self.press = press
        self.conc = conc
        self.v0 = v0
        self.linear = linear

        if isinstance(output, dict):
            self._get_rrho_thermo(mol=output["mol"],
                                  mult=output["mult"],
                                  sigma_r=self.sigma_r,
                                  frequencies_in=output["frequencies"],
                                  elec_energy=output["elec_energy"])

    def get_avg_mom_inertia(self, mol):
        """
        Caclulate the average moment of inertia of a molecule
        :param mol: Molecule
        :return: average moment of inertia, eigenvalues of inertia tensor
        """
        centered_mol = mol.get_centered_molecule()
        inertia_tensor = np.zeros((3, 3))
        for site in centered_mol:
            c = site.coords
            wt = site.specie.atomic_mass
            for i in range(3):
                inertia_tensor[i, i] += wt * (
                        c[(i + 1) % 3] ** 2 + c[(i + 2) % 3] ** 2
                )
            for i, j in [(0, 1), (1, 2), (0, 2)]:
                inertia_tensor[i, j] += -wt * c[i] * c[j]
                inertia_tensor[j, i] += -wt * c[j] * c[i]

        inertia_eigenvals = np.multiply(
            np.linalg.eig(inertia_tensor)[0],
            amu2kg * 1E-20).tolist()  # amuangs^2 to kg m^2

        iav = np.average(inertia_eigenvals)

        return iav, inertia_eigenvals

    def _get_rrho_thermo(self, mol, mult, sigma_r, frequencies_in,
                              elec_energy):
        """
        Caclulate Quasi-RRHO thermochemistry
        :param mol: Molecule
        :param mult: Spin multiplicity
        :param sigma_r: Rotational symmetry number
        :param frequencies: List of frequencies [a.u.]
        :param elec_energy: Electronic energy [a.u.]
        """

        # Calculate mass in kg
        mass = 0
        for site in mol.sites:
            mass += site.specie.atomic_mass
        mass *= amu2kg

        frequencies = list()
        frequencies_shifted = list()
        for freq in frequencies_in:
            # In cm^-1
            if freq < 0:
                continue
            elif freq < 100:
                frequencies.append(freq)
                frequencies_shifted.append(100)
            else:
                frequencies.append(freq)
                frequencies_shifted.append(freq)

        # Get properties related to rotational symmetry. Bav is average moment of inertia
        Bav, i_eigen = self.get_avg_mom_inertia(mol)

        # ZPE
        zpe = sum([0.5 * h * f * c for f in frequencies]) * j2hartree
        self.zpe = zpe

        # Translational component of entropy and energy
        qt = (2 * np.pi * mass * kb * self.temp / (h * h)) ** (
                3 / 2) * kb * self.temp / self.press
        st = R * (np.log(qt) + 5 / 2)
        et = 3 * R * self.temp / 2

        # Electronic component of Entropy
        se = R * np.log(mult)

        # Rotational component of Entropy and Energy
        if self.linear:
            i = np.amax(i_eigen)
            qr = 8 * np.pi ** 2 * i * kb * self.temp / (sigma_r * (h * h))
            sr = R * (np.log(qr) + 1)
            er = R * self.temp
        else:
            qr = np.sqrt(np.pi * i_eigen[0] * i_eigen[1] * i_eigen[2]) / sigma_r * (
                    8 * np.pi ** 2 * kb * self.temp / (h * h)
            ) ** (1.5)
            sr = R * (np.log(qr) + 3 / 2)
            er = 3 * R * self.temp / 2

        # Vibrational component of Entropy and Energy
        ev = 0
        sv = 0
        for f in frequencies:
            ev += h * f * c / (kb * (np.exp(h * f * c / (kb * self.temp)) - 1))
        for f in frequencies_shifted:
            sv += h * f * c / (kb * self.temp * (np.exp(h * f * c / (kb * self.temp)) - 1)) - np.log(1 - np.exp(-1 * h * f * c / (kb * self.temp)))

        sv *= R
        ev *= R

        etot =  (et + er + ev) * kcal2hartree / 1000 + self.zpe
        self.energy_RRHO = etot
        h_corrected = etot + R * self.temp * kcal2hartree / 1000
        self.enthalpy_RRHO = h_corrected

        molarity_corr = 0.000003166488448771253 * self.temp * np.log(
            0.082057338 * self.temp * self.conc)
        self.entropy_RRHO = (st + sr + sv + se) * kcal2hartree / 1000
        self.free_energy_RRHO = (elec_energy + h_corrected -
                                 (self.temp * self.entropy_RRHO))

        self.concentration_corrected_g_RRHO = (self.free_energy_RRHO + molarity_corr)