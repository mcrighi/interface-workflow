# -*- coding: utf-8 -*-
from aiida.backends.utils import load_dbenv, is_dbenv_loaded

__copyright__ = u"Copyright (c), This file is part of the AiiDA platform. For further information please visit http://www.aiida.net/. All rights reserved."
__license__ = "MIT license, see LICENSE.txt file."
__authors__ = "The AiiDA team."
__version__ = "0.7.0"

if not is_dbenv_loaded():
    load_dbenv()

import ase
import math
from aiida.orm import load_node
from aiida.orm import DataFactory
from aiida.orm.code import Code
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array import ArrayData
from aiida.work.db_types import BaseType, Str, Float, Int, Bool
#from aiida.orm.data.base import BaseType, NumericType, Str, Float, Int
from aiida.orm.data.base import NumericType
from aiida.orm.calculation.job.quantumespresso.pw import PwCalculation
from aiida.orm.calculation.job.quantumespresso.pp import PpCalculation
from aiida.work.run import run, submit, async
from aiida.work.workchain import (WorkChain, ToContext, while_)
from aiida.work.workfunction import workfunction
from aiida.work.defaults import registry

from matplotlib import use
use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import math as m
import scipy.integrate as intgrt
from scipy.interpolate import interp1d, CubicSpline 
import os, os.path, time, json
import numpy as np

GPa_to_eV_over_ang3 = 1. / 160.21766208
eVA2_to_Jm2 = 16.0218
Bohr_to_Angstrom = 0.529177
vacuum_slab = 12                     #Vacuum of the slab calculation for the surface energy
iter_adh = 0
iter_adh_x = 0
iter_adh_y = 0
inter_factor = 0.75

# Set up the factories
ParameterData = DataFactory("parameter")
KpointsData = DataFactory("array.kpoints")
PwProcess = PwCalculation.process()

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was

@workfunction
def make_initial_kpts(structure,Nkpts_max,Gamma_Centered):
	"""
	Returns a kpoints array object for a given structure with a constant
	distance between kpoints in reciprocal space on all axis. 
	This distance is computed from the input Nkpts_max, which sets the
	number of Kpoints on the longest reciprocal vector.
	Gamma centered meshes can be selected by setting the Gamma_Centered 
	paramter to Str("True")
	"""
	#compute distance:
	min_vec=min(structure.cell_lengths)
	dist=Float(min_vec*Nkpts_max**(-1))
	#set up kpoints object
	kpoints = KpointsData()
	kpoints.set_cell_from_structure(structure)
	if Gamma_Centered == "True":
		offset=[0., 0., 0.]
	else:
		offset=[0.5, 0.5, 0.5]
	kpoints.set_kpoints_mesh_from_density(float(dist), offset,force_parity=False)
	Kpoints_Data={'kpoints': kpoints, 'distance': dist}
	return Kpoints_Data

@workfunction
def make_kpts(structure,distance,Gamma_Centered):
    """
    Returns a kpoints array object for a given structure with a constant
    distance between kpoints in reciprocal space on all axis. 
    Gamma centered meshes can be selected by setting the Gamma_Centered 
    paramter to True
    """
    #set up kpoints object
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    if Gamma_Centered == True:
    	offset=[0., 0., 0.]
    else:
    	offset=[0.5, 0.5, 0.5]
    kpoints.set_kpoints_mesh_from_density(float(distance), offset,force_parity=False)
    return kpoints


def get_pseudos(structure, family_name):
    """
    Set the pseudo to use for all atomic kinds, picking pseudos from the
    family with name family_name.

    :note: The structure must already be set.

    :param family_name: the name of the group containing the pseudos
    """
    from collections import defaultdict
    from aiida.orm.data.upf import get_pseudos_from_structure

    # A dict {kind_name: pseudo_object}
    kind_pseudo_dict = get_pseudos_from_structure(structure, family_name)

    # We have to group the species by pseudo, I use the pseudo PK
    # pseudo_dict will just map PK->pseudo_object
    pseudo_dict = {}
    # Will contain a list of all species of the pseudo with given PK
    pseudo_species = defaultdict(list)

    for kindname, pseudo in kind_pseudo_dict.iteritems():
        pseudo_dict[pseudo.pk] = pseudo
        pseudo_species[pseudo.pk].append(kindname)

    pseudos = {}
    for pseudo_pk in pseudo_dict:
        pseudo = pseudo_dict[pseudo_pk]
        kinds = pseudo_species[pseudo_pk]
        for kind in kinds:
            pseudos[kind] = pseudo

    return pseudos


def generate_cube_input_params(structure, codename, pseudo_family, metal, magnet, vdw, scf_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling):
    # The inputs
    inputs = PwCalculation.process().get_inputs_template()

    # The structure
    inputs.structure = structure

    #Hydra settings
    inputs.code = Code.get_from_string(codename.value)
    inputs._options.resources = {"num_machines": 1}
    inputs._options.max_wallclock_seconds = scf_wallclock_max_time
    inputs._options.queue_name = "s3par6c"

    # Kpoints
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints_mesh = 20
    kpoints.set_kpoints_mesh([kpoints_mesh, kpoints_mesh, kpoints_mesh])
    inputs.kpoints = kpoints

    # Calculation parameters
    parameters_dict = {
        "CONTROL": {"calculation": "scf",
                    "tstress": True,  #  Important that this stays to get stress
                    "tprnfor": True,},
        "SYSTEM": {"ecutwfc": energy_cutoff_wavefunction,
                   "ecutrho": energy_cutoff_charge
                  },
        "ELECTRONS": {"conv_thr": 1.e-6,}
    }

    if metal == True:
       parameters_dict["SYSTEM"]["occupations"] = "smearing"
       parameters_dict["SYSTEM"]["smearing"] = "gaussian"
       parameters_dict["SYSTEM"]["degauss"] = 0.02

    if magnet == True:
       parameters_dict["SYSTEM"]["nspin"] = 2
       parameters_dict["SYSTEM"]["starting_magnetization"] = 0.6

    if vdw == True:
       parameters_dict["SYSTEM"]["vdw_corr"] = "DFT-D"
       parameters_dict["SYSTEM"]["london_s6"] = vdW_scaling

    ParameterData = DataFactory("parameter")
    inputs.parameters = ParameterData(dict=parameters_dict)

    # Pseudopotentials
    inputs.pseudo = get_pseudos(structure, str(pseudo_family))

    return inputs

def generate_bulk_input_kpoints(structure, codename, pseudo_family, metal, magnet, vdw, kpoints, relax_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling):
    # The inputs
    inputs = PwCalculation.process().get_inputs_template()

    # The structure
    inputs.structure = structure

    #Hydra settings
    inputs.code = Code.get_from_string(codename.value)
    inputs._options.resources = {"num_machines": 1}
    inputs._options.max_wallclock_seconds = relax_wallclock_max_time
    inputs._options.queue_name = "s3par6c"

    # Old Kpoints mesh, without convergence
    """
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints_mesh = 6
    kpoints.set_kpoints_mesh([kpoints_mesh, kpoints_mesh, kz])
    """
    inputs.kpoints = kpoints

    # Calculation parameters
    parameters_dict = {
        "CONTROL": {"calculation": "relax",
                    "restart_mode": "from_scratch",
                    "tstress": True,  # Important that this stays to get stress
                    "tprnfor": True,
                    "verbosity": "low"},
        "SYSTEM": {"ecutwfc": energy_cutoff_wavefunction,
                   "ecutrho": energy_cutoff_charge},
        "ELECTRONS": {"electron_maxstep": 200,
                      "mixing_mode": "plain",
                      "mixing_beta": 0.5,
                      "conv_thr": 1.e-6}
    }

    if metal == True:
       parameters_dict["SYSTEM"]["occupations"] = "smearing"
       parameters_dict["SYSTEM"]["smearing"] = "gaussian"
       parameters_dict["SYSTEM"]["degauss"] = 0.02

    if magnet == True:
       parameters_dict["SYSTEM"]["nspin"] = 2
       parameters_dict["SYSTEM"]["starting_magnetization"] = 0.6

    if vdw == True:
       parameters_dict["SYSTEM"]["vdw_corr"] = "DFT-D"
       parameters_dict["SYSTEM"]["london_s6"] = vdW_scaling


    ParameterData = DataFactory("parameter")
    inputs.parameters = ParameterData(dict=parameters_dict)

    # Pseudopotentials
    inputs.pseudo = get_pseudos(structure, str(pseudo_family))

    return inputs


def generate_bulk_input_params(structure, codename, pseudo_family, metal, magnet, vdw, kdistance, relax_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling):
    # The inputs
    inputs = PwCalculation.process().get_inputs_template()

    # The structure
    inputs.structure = structure

    #Hydra settings
    inputs.code = Code.get_from_string(codename.value)
    inputs._options.resources = {"num_machines": 1}
    inputs._options.max_wallclock_seconds = relax_wallclock_max_time
    inputs._options.queue_name = "s3par6c"

    # Old Kpoints mesh, without convergence
    """
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints_mesh = 6
    kpoints.set_kpoints_mesh([kpoints_mesh, kpoints_mesh, kz])
    inputs.kpoints = kpoints
    """

    # New Kpoints mesh, with convergence test
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    offset=[0.5, 0.5, 0.5]
    kpoints.set_kpoints_mesh_from_density(float(kdistance), offset,force_parity=False)
    inputs.kpoints = kpoints

    # Calculation parameters
    parameters_dict = {
        "CONTROL": {"calculation": "relax",
                    "restart_mode": "from_scratch",
                    "tstress": True,  # Important that this stays to get stress
                    "tprnfor": True,
                    "verbosity": "low"},
        "SYSTEM": {"ecutwfc": energy_cutoff_wavefunction,
                   "ecutrho": energy_cutoff_charge},
        "ELECTRONS": {"electron_maxstep": 200,
                      "mixing_mode": "plain",
                      "mixing_beta": 0.5,
                      "conv_thr": 1.e-6}
    }

    if metal == True:
       parameters_dict["SYSTEM"]["occupations"] = "smearing"
       parameters_dict["SYSTEM"]["smearing"] = "gaussian"
       parameters_dict["SYSTEM"]["degauss"] = 0.02

    if magnet == True:
       parameters_dict["SYSTEM"]["nspin"] = 2
       parameters_dict["SYSTEM"]["starting_magnetization"] = 0.6

    if vdw == True:
       parameters_dict["SYSTEM"]["vdw_corr"] = "DFT-D"
       parameters_dict["SYSTEM"]["london_s6"] = vdW_scaling


    ParameterData = DataFactory("parameter")
    inputs.parameters = ParameterData(dict=parameters_dict)

    # Pseudopotentials
    inputs.pseudo = get_pseudos(structure, str(pseudo_family))

    return inputs


def generate_scf_input_params(structure, codename, pseudo_family, metal, magnet, bands, vdw, kdistance, scf_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling):
    # The inputs
    inputs = PwCalculation.process().get_inputs_template()

    # The structure
    inputs.structure = structure

    #Hydra settings
    inputs.code = Code.get_from_string(codename.value)
    inputs._options.resources = {"num_machines": 1}
    inputs._options.max_wallclock_seconds = scf_wallclock_max_time
    inputs._options.queue_name = "s3par6c"

    # New Kpoints mesh, with convergence test
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    offset=[0.5, 0.5, 0.5]
    kpoints.set_kpoints_mesh_from_density(float(kdistance), offset,force_parity=False)
    inputs.kpoints = kpoints

    # Calculation parameters
    parameters_dict = {
        "CONTROL": {"calculation": "scf",
                    "restart_mode": "from_scratch",
                    "tstress": True,  # Important that this stays to get stress
                    "tprnfor": True,
                    "verbosity": "low",
                    "wf_collect": True},
        "SYSTEM": {"ecutwfc": energy_cutoff_wavefunction,
                   "ecutrho": energy_cutoff_charge},
        "ELECTRONS": {"electron_maxstep": 300,
                      "mixing_mode": "plain",
                      "mixing_beta": 0.25,
                      "conv_thr": 1.e-6}
    }

    if metal == True:
       parameters_dict["SYSTEM"]["occupations"] = "smearing"
       parameters_dict["SYSTEM"]["smearing"] = "gaussian"
       parameters_dict["SYSTEM"]["degauss"] = 0.02

    if magnet == True:
       parameters_dict["SYSTEM"]["nspin"] = 2
       parameters_dict["SYSTEM"]["starting_magnetization"] = 0.6

    if bands is not None:
       parameters_dict["SYSTEM"]["nbnd"] = bands

    if vdw == True:
       parameters_dict["SYSTEM"]["vdw_corr"] = "DFT-D"
       parameters_dict["SYSTEM"]["london_s6"] = vdW_scaling

    ParameterData = DataFactory("parameter")
    inputs.parameters = ParameterData(dict=parameters_dict)

    # Pseudopotentials
    inputs.pseudo = get_pseudos(structure, str(pseudo_family))

    return inputs


def generate_scfinter_input_params(structure, codename, pseudo_family, metal, magnet, bands, vdw, kdistance, scf_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling):
    # The inputs
    inputs = PwCalculation.process().get_inputs_template()

    # The structure
    inputs.structure = structure

    #Hydra settings
    inputs.code = Code.get_from_string(codename.value)
    inputs._options.resources = {"num_machines": 1}
    inputs._options.max_wallclock_seconds = scf_wallclock_max_time
    inputs._options.queue_name = "s3par6c"

    # New Kpoints mesh, with convergence test
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    offset=[0.5, 0.5, 0.5]
    kpoints.set_kpoints_mesh_from_density(float(kdistance), offset,force_parity=False)
    inputs.kpoints = kpoints

    # Calculation parameters
    parameters_dict = {
        "CONTROL": {"calculation": "scf",
                    "restart_mode": "from_scratch",
                    "tstress": True,  # Important that this stays to get stress
                    "tprnfor": True,
                    "verbosity": "low",
                    "wf_collect": True},
        "SYSTEM": {"ecutwfc": energy_cutoff_wavefunction,
                   "ecutrho": energy_cutoff_charge},
        "ELECTRONS": {"electron_maxstep": 300,
                      "mixing_mode": "plain",
                      "mixing_beta": 0.25,
                      "conv_thr": 1.e-6}
    }

    if metal == True:
       parameters_dict["SYSTEM"]["occupations"] = "smearing"
       parameters_dict["SYSTEM"]["smearing"] = "gaussian"
       parameters_dict["SYSTEM"]["degauss"] = 0.02

    if magnet == True:
       parameters_dict["SYSTEM"]["nspin"] = 2
       parameters_dict["SYSTEM"]["starting_magnetization"] = 0.6

    if bands is not None:
       nbands = bands * 2
       parameters_dict["SYSTEM"]["nbnd"] = nbands

    if vdw == True:
       parameters_dict["SYSTEM"]["vdw_corr"] = "DFT-D"
       parameters_dict["SYSTEM"]["london_s6"] = vdW_scaling

    ParameterData = DataFactory("parameter")
    inputs.parameters = ParameterData(dict=parameters_dict)

    # Pseudopotentials
    inputs.pseudo = get_pseudos(structure, str(pseudo_family))

    return inputs


def generate_slab_input_params(structure, codename, pseudo_family, metal, magnet, bands, vdw, kdistance, relax_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling):
    # The inputs
    inputs = PwCalculation.process().get_inputs_template()

    # The structure
    inputs.structure = structure

    #Hydra settings
    inputs.code = Code.get_from_string(codename.value)
    inputs._options.resources = {"num_machines": 1}
    inputs._options.max_wallclock_seconds = relax_wallclock_max_time
    inputs._options.queue_name = "s3par6c"

    """
    # Kpoints
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints_mesh = 6
    kpoints.set_kpoints_mesh([kpoints_mesh, kpoints_mesh, 1])
    inputs.kpoints = kpoints
    """

    # New Kpoints mesh, with convergence test
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    offset=[0.5, 0.5, 0.5]
    kpoints.set_kpoints_mesh_from_density(float(kdistance), offset,force_parity=False)
    inputs.kpoints = kpoints

    # Calculation parameters
    parameters_dict = {
        "CONTROL": {"calculation": "relax",
                    "restart_mode": "from_scratch",
                    "tstress": True,  # Important that this stays to get stress
                    "tprnfor": True,
                    "verbosity": "low",
                    "wf_collect" : True},
        "SYSTEM": {"ecutwfc": energy_cutoff_wavefunction,
                   "ecutrho": energy_cutoff_charge},
        "ELECTRONS": {"electron_maxstep": 300,
                      "mixing_mode": "plain",
                      "mixing_beta": 0.25,
                      "conv_thr": 1.e-6}
    }

    if metal == True:
       parameters_dict["SYSTEM"]["occupations"] = "smearing"
       parameters_dict["SYSTEM"]["smearing"] = "gaussian"
       parameters_dict["SYSTEM"]["degauss"] = 0.02

    if magnet == True:
       parameters_dict["SYSTEM"]["nspin"] = 2
       parameters_dict["SYSTEM"]["starting_magnetization"] = 0.6

    if bands is not 0:
       parameters_dict["SYSTEM"]["nbnd"] = bands

    if vdw == True:
       parameters_dict["SYSTEM"]["vdw_corr"] = "DFT-D"
       parameters_dict["SYSTEM"]["london_s6"] = vdW_scaling

    ParameterData = DataFactory("parameter")
    inputs.parameters = ParameterData(dict=parameters_dict)

    # Pseudopotentials
    inputs.pseudo = get_pseudos(structure, str(pseudo_family))

    return inputs


def generate_inter_input_params(structure, codename, pseudo_family, metal, magnet, bands, vdw, kdistance, relax_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling):
    # The inputs
    inputs = PwCalculation.process().get_inputs_template()

    # The structure
    inputs.structure = structure

    #Hydra settings
    inputs.code = Code.get_from_string(codename.value)
    inputs._options.resources = {"num_machines": 1}
    inputs._options.max_wallclock_seconds = relax_wallclock_max_time
    inputs._options.queue_name = "s3par6c"

    """
    # Kpoints
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints_mesh = 6
    kpoints.set_kpoints_mesh([kpoints_mesh, kpoints_mesh, 1])
    inputs.kpoints = kpoints
    """

    # New Kpoints mesh, with convergence test
    KpointsData = DataFactory("array.kpoints")
    kpoints = KpointsData()
    kpoints.set_cell_from_structure(structure)
    offset=[0.5, 0.5, 0.5]
    kpoints.set_kpoints_mesh_from_density(float(kdistance), offset,force_parity=False)
    inputs.kpoints = kpoints

    # Calculation parameters
    parameters_dict = {
        "CONTROL": {"calculation": "relax",
                    "restart_mode": "from_scratch",
                    "nstep": 200,
                    "tstress": True,  # Important that this stays to get stress
                    "tprnfor": True,
                    "verbosity": "low",
                    "wf_collect" : True},
        "SYSTEM": {"ecutwfc": energy_cutoff_wavefunction,
                   "ecutrho": energy_cutoff_charge},
        "ELECTRONS": {"electron_maxstep": 500,
                      "mixing_mode": "plain",
                      "mixing_beta": 0.2,
                      "conv_thr": 1.e-6}
    }

    if metal == True:
       parameters_dict["SYSTEM"]["occupations"] = "smearing"
       parameters_dict["SYSTEM"]["smearing"] = "gaussian"
       parameters_dict["SYSTEM"]["degauss"] = 0.02

    if magnet == True:
       parameters_dict["SYSTEM"]["nspin"] = 2
       parameters_dict["SYSTEM"]["starting_magnetization"] = 0.6

    if bands is not None:
       nbands = bands * 2
       parameters_dict["SYSTEM"]["nbnd"] = nbands

    if vdw == True:
       parameters_dict["SYSTEM"]["vdw_corr"] = "DFT-D"
       parameters_dict["SYSTEM"]["london_s6"] = vdW_scaling

    #N_atoms = len(structure.sites)
    N_atoms_inter = structure.get_ase().get_number_of_atoms()

    list_fixed = [[True, True, False] for i in np.arange(N_atoms_inter)]

    settings_dict = {'fixed_coords': list_fixed,}

    ParameterData = DataFactory("parameter")
    inputs.parameters = ParameterData(dict=parameters_dict)
    inputs.settings = ParameterData(dict=settings_dict)

    # Pseudopotentials
    inputs.pseudo = get_pseudos(structure, str(pseudo_family))

    return inputs


def get_first_deriv(stress):
    """
    Return the energy first derivative from the stress
    """
    from numpy import trace
    # Get the pressure (GPa)
    p = trace(stress) / 3.
    # Pressure is -dE/dV; moreover p in kbar, we need to convert
    # it to eV/angstrom^3 to be consisten
    dE = -p * GPa_to_eV_over_ang3
    return dE


def get_volume_energy_and_derivative(output_parameters):
    """
    Given the output parameters of the pw calculation,
    return the volume (ang^3), the energy (eV), and the energy
    derivative (eV/ang^3)
    """
    V = output_parameters.dict.volume
    E = output_parameters.dict.energy
    dE = get_first_deriv(output_parameters.dict.stress)

    return (V, E, dE)


def get_second_derivative(outp1, outp2):
    """
    Given the output parameters of the two pw calculations,
    return the second derivative obtained from finite differences
    from the pressure of the two calculations (eV/ang^6)
    """
    dE1 = get_first_deriv(outp1.dict.stress)
    dE2 = get_first_deriv(outp2.dict.stress)
    V1 = outp1.dict.volume
    V2 = outp2.dict.volume
    return (dE2 - dE1) / (V2 - V1)


def get_abc(V, E, dE, ddE):
    """
    Given the volume, energy, energy first derivative and energy
    second derivative, return the a,b,c coefficients of
    a parabola E = a*V^2 + b*V + c
    """
    a = ddE / 2.
    b = dE - ddE * V
    c = E - V * dE + V ** 2 * ddE / 2.

    return a, b, c


def get_new_structure(original_structure, new_volume):
    """
    Given a structure and a new volume, rescale the structure to the new volume
    """
    initial_volume = original_structure.get_cell_volume()
    scale_factor = (new_volume / initial_volume) ** (1. / 3.)
    scaled_structure = rescale(original_structure, Float(scale_factor))
    return scaled_structure


def get_energy(output_parameters):
    """
    Given the output parameters of the pw calculation,
    return the energy (eV).
    """
    E = output_parameters.dict.energy

    return (E)


def replica_for_pes_100(replica, support, array_accumulo, alat_x, alat_y):
    
    replica[1,2] = support[1,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[1,1] = support[1,1]+alat_x*0.25
    replica[1,2] = support[1,2]-alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[1,1] = support[1,1]+alat_x*0.75
    replica[1,2] = support[1,2]-alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[2,1] = support[2,1]+alat_x*0.5
    replica[2,2] = support[2,2]-alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[2,:]])
    replica[3,2] = support[3,2]+alat_y*0.75
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[3,1] = support[3,1]+alat_x*0.75
    replica[3,2] = support[3,2]
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[3,1] = support[3,1]+alat_x*0.75
    replica[3,2] = support[3,2]+alat_y*0.75
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[4,2] = support[4,2]+alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.75
    replica[4,2] = support[4,2]
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.75
    replica[4,2] = support[4,2]+alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.25
    replica[4,2] = support[4,2]-alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.25
    replica[4,2] = support[4,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.5
    replica[4,2] = support[4,2]-alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.5
    replica[4,2] = support[4,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[5,2] = support[5,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[5,:]])
    replica[5,1] = support[5,1]+alat_x*0.5
    replica[5,2] = support[5,2]
    array_accumulo = np.vstack([array_accumulo, replica[5,:]])
    replica[5,1] = support[5,1]+alat_x*0.5
    replica[5,2] = support[5,2]+alat_x*0.5
    array_accumulo = np.vstack([array_accumulo, replica[5,:]])
    replica[6,1] = support[6,1]+alat_x*0.5
    array_accumulo = np.vstack([array_accumulo, replica[6,:]])
    replica[6,1] = support[6,1]+alat_x*0.25
    replica[6,2] = support[6,2]-alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[6,:]])
    replica[6,1] = support[6,1]+alat_x*0.25
    replica[6,2] = support[6,2]+alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[6,:]])
    replica[7,2] = support[7,2]+alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[7,:]])
    replica[7,1] = support[7,1]+alat_x*0.25
    replica[7,2] = support[7,2]
    array_accumulo = np.vstack([array_accumulo, replica[7,:]])
    replica[7,1] = support[7,1]+alat_x*0.25
    replica[7,2] = support[7,2]+alat_x*0.25
    array_accumulo = np.vstack([array_accumulo, replica[7,:]])
    
    return array_accumulo


def replica_for_pes_dia100(replica, support, array_accumulo, alat_x, alat_y):
    
    replica[1,2] = support[1,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[3,2] = support[3,2]+alat_y*0.75
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[3,1] = support[3,1]+alat_x*0.75
    replica[3,2] = support[3,2]
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[3,1] = support[3,1]+alat_x*0.75
    replica[3,2] = support[3,2]+alat_y*0.75
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[4,2] = support[4,2]+alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.75
    replica[4,2] = support[4,2]
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.75
    replica[4,2] = support[4,2]+alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[5,1] = support[5,1]+alat_x*0.5
    array_accumulo = np.vstack([array_accumulo, replica[5,:]])
    replica[6,2] = support[6,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[6,:]])
    replica[6,1] = support[6,1]+alat_x*0.5
    replica[6,2] = support[6,2]
    array_accumulo = np.vstack([array_accumulo, replica[6,:]])
    replica[6,1] = support[6,1]+alat_x*0.5
    replica[6,2] = support[6,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[6,:]])
    replica[7,1] = support[7,1]+alat_x*0.5
    array_accumulo = np.vstack([array_accumulo, replica[7,:]])
    replica[8,2] = support[8,2]+alat_y*0.75
    array_accumulo = np.vstack([array_accumulo, replica[8,:]])
    replica[8,1] = support[8,1]+alat_x*0.25
    replica[8,2] = support[8,2]
    array_accumulo = np.vstack([array_accumulo, replica[8,:]])
    replica[8,1] = support[8,1]+alat_x*0.25
    replica[8,2] = support[8,2]+alat_y*0.75
    array_accumulo = np.vstack([array_accumulo, replica[8,:]])
    replica[9,2] = support[9,2]+alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[9,:]])
    replica[9,1] = support[9,1]+alat_x*0.25
    replica[9,2] = support[9,2]
    array_accumulo = np.vstack([array_accumulo, replica[9,:]])
    replica[9,1] = support[9,1]+alat_x*0.25
    replica[9,2] = support[9,2]+alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[9,:]])
    replica[11,2] = support[11,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[11,:]])

    return array_accumulo


def replica_for_pes_111(replica, support, array_accumulo, alat_x, alat_y):
    
    replica[0,1] = support[0,1]+alat_x*0.5
    replica[0,2] = support[0,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[0,:]])
    replica[1,1] = support[1,1]+alat_x*0.25
    replica[1,2] = support[1,2]+alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[1,1] = support[1,1]+alat_x*0.75
    replica[1,2] = support[1,2]+alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[1,1] = support[1,1]+alat_x*0.25
    replica[1,2] = support[1,2]+alat_y*0.75
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[1,1] = support[1,1]+alat_x*0.75
    replica[1,2] = support[1,2]+alat_y*0.75
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[1,1] = support[1,1]+alat_x*0.5
    replica[1,2] = support[1,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[2,1] = support[2,1]+alat_x*0.5
    replica[2,2] = support[2,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[2,:]])
    replica[3,1] = support[3,1]+alat_x*0.5
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[3,1] = support[3,1]+alat_x*0.25
    replica[3,2] = support[3,2]+alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[3,1] = support[3,1]-alat_x*0.25
    replica[3,2] = support[3,2]+alat_y*0.75
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[3,1] = support[3,1]
    replica[3,2] = support[3,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[3,1] = support[3,1]+alat_x*0.5
    replica[3,2] = support[3,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[4,1] = support[4,1]-alat_x*0.25
    replica[4,2] = support[4,2]+alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.25
    replica[4,2] = support[4,2]-alat_y*0.25
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.5
    replica[4,2] = support[4,2]
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]
    replica[4,2] = support[4,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.5
    replica[4,2] = support[4,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[5,1] = support[5,1]-alat_x*0.5
    replica[5,2] = support[5,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[5,:]])
    
    return array_accumulo

    
def replica_for_pes_fcc110(replica, support, array_accumulo, alat_x, alat_y):

    replica[1,2] = support[1,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[3,1] = support[3,1]+alat_x*0.5
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[4,1] = support[4,1]
    replica[4,2] = support[4,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.5
    replica[4,2] = support[4,2]
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.5
    replica[4,2] = support[4,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[5,1] = support[5,1]+alat_x*0.5
    array_accumulo = np.vstack([array_accumulo, replica[5,:]])
    replica[7,2] = support[7,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[7,:]])
    
    return array_accumulo

def replica_for_pes_bcc110(replica, support, array_accumulo, alat_x, alat_y):

    replica[0,1] = support[0,1]+alat_x*0.5
    replica[0,2] = support[0,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[0,:]])
    replica[1,1] = support[1,1]
    replica[1,2] = support[1,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[1,1] = support[1,1]+alat_x*0.5
    replica[1,2] = support[1,2]
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[1,1] = support[1,1]+alat_x*0.5
    replica[1,2] = support[1,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[2,1] = support[2,1]
    replica[2,2] = support[2,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[2,:]])
    replica[2,1] = support[2,1]+alat_x*0.5
    replica[2,2] = support[2,2]
    array_accumulo = np.vstack([array_accumulo, replica[2,:]])
    replica[2,1] = support[2,1]+alat_x*0.5
    replica[2,2] = support[2,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[2,:]])
    replica[3,1] = support[3,1]+alat_x*0.5
    replica[3,2] = support[3,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[4,1] = support[4,1]
    replica[4,2] = support[4,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.5
    replica[4,2] = support[4,2]
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.5
    replica[4,2] = support[4,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    
    return array_accumulo


def replica_for_pes_dia110(replica, support, array_accumulo, alat_x, alat_y):

    replica[0,1] = support[0,1]+alat_x*0.5
    replica[0,2] = support[0,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[0,:]])
    replica[1,1] = support[1,1]
    replica[1,2] = support[1,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[1,1] = support[1,1]+alat_x*0.5
    replica[1,2] = support[1,2]
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[1,1] = support[1,1]+alat_x*0.5
    replica[1,2] = support[1,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[1,:]])
    replica[2,1] = support[2,1]+alat_x*0.5
    replica[2,2] = support[2,2]-alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[2,:]])
    replica[3,1] = support[3,1]+alat_x*0.5
    replica[3,2] = support[3,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[3,:]])
    replica[4,1] = support[4,1]
    replica[4,2] = support[4,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.5
    replica[4,2] = support[4,2]
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[4,1] = support[4,1]+alat_x*0.5
    replica[4,2] = support[4,2]+alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[4,:]])
    replica[5,1] = support[5,1]+alat_x*0.5
    replica[5,2] = support[5,2]-alat_y*0.5
    array_accumulo = np.vstack([array_accumulo, replica[5,:]])
    
    return array_accumulo


@workfunction
def rescale(structure, scale):
    """
    Workfunction to rescale a structure

    :param structure: An AiiDA structure to rescale
    :param scale: The scale factor
    :return: The rescaled structure
    """
    the_ase = structure.get_ase()
    new_ase = the_ase.copy()
    new_ase.set_cell(the_ase.get_cell() * float(scale), scale_atoms=True)
    new_structure = DataFactory('structure')(ase=new_ase)
    return new_structure

@workfunction
def optalat(structure):
    """
    Output optimized lattice parameter
    """
    from aiida.orm import DataFactory
    opt_alat = 2*Float(structure.cell[0][1])
    return opt_alat

@workfunction
def optalat_hex(structure):
    """
    Output optimized lattice parameter
    """
    from aiida.orm import DataFactory
    opt_alat = Float(structure.cell[0][0])
    return opt_alat

@workfunction
def surfen(E_slab, E_bulk, N_ratio, area):
    """
    Output surface energy of the system
    """
    from aiida.orm import DataFactory
    E_surf = (Float(E_slab) - Float(N_ratio) * Float(E_bulk)) * math.pow(2. * Float(area), -1) * eVA2_to_Jm2
    return E_surf

@workfunction
def interen(E_inter, E_slab, area):
    """
    Output adhesion energy of the interface
    """
    from aiida.orm import DataFactory
    E_inter = (Float(E_inter) - 2 * Float(E_slab)) * math.pow(Float(area), -1) * eVA2_to_Jm2
    return E_inter

@workfunction
def workofsep(w_sep):
    """
    Output work of separation
    """
    from aiida.orm import DataFactory
    wsep = Float(w_sep) 
    return wsep

@workfunction
def shear(shear_strength):
    """
    Output ideal shear strength
    """
    from aiida.orm import DataFactory
    shearstrength = Float(shear_strength)*10.0
    return shearstrength

@workfunction
def get_energy_test(structure, kpoints, code_name, pseudo_family, metal, magnet, vdw, relax_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling):
	"""
	Run a static QE calculation to get the total energy of the system
	"""
	JobCalc=PwCalculation.process()
	inputs=generate_bulk_input_kpoints(structure, code_name, pseudo_family, bool(metal), bool(magnet), bool(vdw), kpoints, int(relax_wallclock_max_time), energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling)
	result = run(JobCalc,**inputs)
	E=Float(result['output_parameters'].dict.energy)
	return E

@workfunction
def converge_kpts(structure, tolerance_per_atom, code_name, pseudo_family, metal, magnet, vdw, relax_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling):
	"""
	Converge the total energy with respect to kpoints for a given structure
	to the given tolerace. The tolerance must be given in eV/atom. The return is
	a the converged minimum distance between kpoints in reciprocal space and can be
	used as an input alongside any structure with the make_kpts() function.
	"""
	NrAtoms=structure.get_ase().get_number_of_atoms()
	tol=float(tolerance_per_atom)*NrAtoms #actual tolerance in total energy
	#initialize a dictionary where all important infos of the convergence run are stored:
	convergence_info={}
	convergence_info['convergence_parameters']={'tolerance_per_atom': tolerance_per_atom, 'total_convergence': tol}
	#initial_calculation:
	MaxNrKpts=2.0
	initial_data=make_initial_kpts(structure, Float(MaxNrKpts), Str('False'))
	kpts=initial_data['kpoints']
	dist=initial_data['distance']
	mesh=kpts.get_attrs()['mesh']
	E=get_energy_test(structure, kpts, code_name, pseudo_family, metal, magnet, vdw, relax_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling)
	Delta_E=abs(float(E))
	#store initial values in dict:
	convergence_info['step0']={'kpoints': kpts, 'dist': dist, 'mesh': mesh, 'energy': E, 'DeltaE': Float(Delta_E), 'DeltaE_per_atom': Float(Delta_E/NrAtoms)}
	#initialize loop for convergence:
	reduction=0.9 #10% reduction of distance in each step
	i=0
	E_old=E
	mesh_old=mesh
	#converge to the required accuracy in total energy:
	while float(Delta_E) > float(tol):
		dist=float(dist)*reduction
		kpts=make_kpts(structure, Float(dist),Str('False'))
		#check if the reduced density actually results in a new mesh before calculating E:
		mesh=kpts.get_attrs()['mesh']
		if not mesh == mesh_old:
			i=i+1
			E=get_energy_test(structure, kpts, code_name, pseudo_family, metal, magnet, vdw, relax_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling)
			Delta_E=abs(float(E)-float(E_old))
			convergence_info['step'+str(i)]={'kpoints': kpts, 'dist': Float(dist), 'mesh': mesh, 'Energy': E, 'DeltaE': Float(Delta_E), 'DeltaE_per_atom': Float(Delta_E/NrAtoms)}
			E_old=E
			mesh_old=mesh
	#write converged distance to dict and repeat info of last iteration with key "final_step":
	convergence_info['final_dist']=Float(dist)
	convergence_info['final_step']={'kpoints': kpts, 'dist': Float(dist), 'mesh': mesh, 'Energy': E, 'DeltaE': Float(Delta_E), 'DeltaE_per_atom': Float(Delta_E/NrAtoms)}
	#TBD
	#maybe append convergence_info to the log? Do some other output? Store in ParameterData?
	#TBD

	#return the converged distance
	return Float(dist)


@workfunction
def make_upper_slab(total_slab_structure):
	new=total_slab_structure.copy()
	new.clear_sites()
	sites=total_slab_structure.sites
	zs=[]
	for s in sites:
		z=float(str(s).split()[-1].split(",")[-1]) #get z coordinate of site
		zs.append(z)
	zs=np.asarray(zs)
	middle=np.mean(zs)
	for s in sites:
		z=float(str(s).split()[-1].split(",")[-1]) #get z coordinate of site
		if z>middle:
			new.append_site(s)
	new.sites
	return new


@workfunction	
def make_lower_slab(total_slab_structure):
	new=total_slab_structure.copy()
	new.clear_sites()
	sites=total_slab_structure.sites
	zs=[]
	for s in sites:
		z=float(str(s).split()[-1].split(",")[-1]) #get z coordinate of site
		zs.append(z)
	zs=np.asarray(zs)
	middle=np.mean(zs)
	for s in sites:
		z=float(str(s).split()[-1].split(",")[-1]) #get z coordinate of site
		if z<middle:
			new.append_site(s)
	new.sites
	return new

def get_pp_input_params(parent_folder,code_name):
	inputs  =  PpCalculation.process().get_inputs_template()
	inputs.code  =  Code.get_from_string(str(code_name))
	inputs._options.max_wallclock_seconds  =  10 * 60
	inputs._options.resources  =  {u'num_machines': 1}
        inputs._options.queue_name = "s3par6c"
	
        pp_dict = {u'INPUTPP': {u'plot_num': 0}}
	ParameterData  =  DataFactory('parameter')
	inputs.parameters = ParameterData(dict = pp_dict)
	inputs.parent_folder = parent_folder
	return inputs

	
@workfunction
def run_slab_pp_submit(parent_tot, parent_upp, parent_low, code_name):
	pp_calculations = {}
	parent_pk_list = [parent_tot, parent_upp, parent_low]
	labels = ['tot','upp','low']
	PP_proc = PpCalculation.process()
	for label, parent in zip(labels,parent_pk_list):
		pp_inputs = get_pp_input_params(parent, code_name)
		future = submit(PP_proc, **pp_inputs)
		pp_calculations[label] = load_node(future.pid)
	results = {}
	while pp_calculations['tot'].get_state() != 'FINISHED' and pp_calculations['upp'].get_state() != 'FINISHED' and pp_calculations['low'].get_state() != 'FINISHED':
		#print ('tot state: '+pp_calculations['tot'].get_state()+' upp state: '+pp_calculations['upp'].get_state()+' low state: '+pp_calculations['low'].get_state())
		time.sleep(10)
	#print('-------------')
	#print ('tot state: '+pp_calculations['tot'].get_state()+' upp state: '+pp_calculations['upp'].get_state()+' low state: '+pp_calculations['low'].get_state())
	#print('-------------')
	for label in labels:
		results[label] = Str(pp_calculations[label].out.retrieved.get_abs_path()+'/path/aiida.filplot')
	#Create ParameterData node for output
	Out_PD=ParameterData()
	Out_PD.set_dict(results)
	Out_PD.label = u'charge density files locations'
	Out_PD.description = u'Charge densities of total, upper, and lower slabs have been stored in files at the locations stored in this ParameterData node.'
	return Out_PD	
	
	
@workfunction
def run_slab_pp_async(parent_tot, parent_upp, parent_low, code_name):
	pp_calculations = {}
	parent_pk_list = [parent_tot, parent_upp, parent_low]
	labels = ['tot','upp','low']
	PP_proc = PpCalculation.process()
	for label, parent in zip(labels,parent_pk_list):
		pp_inputs = get_pp_input_params(parent, code_name)
		future = async(PP_proc, **pp_inputs)
		pp_calculations[label] = future
	results = {}
	for label in labels:
		results[label] = Str(pp_calculations[label].result()['retrieved'].get_abs_path()+'/path/aiida.filplot')
	#Create ParameterData node for output
	Out_PD=ParameterData()
	Out_PD.set_dict(results)
	Out_PD.label = u'charge density files locations'
	Out_PD.description = u'Charge densities of total, upper, and lower slabs have been stored in files at the locations stored in this ParameterData node.'
	return Out_PD

@workfunction
def compute_charge_density_difference(ParameterDataNode, plotting):
    CHG_file_tot = str(ParameterDataNode.dict.tot)
    CHG_file_upp = str(ParameterDataNode.dict.upp)
    CHG_file_lwr = str(ParameterDataNode.dict.low)
    try:
        chg_tot = open(CHG_file_tot, "r")
    except IOError:
        print(" ")
        print(" Error opening total charge density file ")
        print(" ")
        quit()
    try:
        chg_upp = open(CHG_file_upp, "r")
    except IOError:
        print(" ")
        print(" Error opening upper charge density file ")
        print(" ")
        quit()
    try:
        chg_lwr = open(CHG_file_lwr, "r")
    except IOError:
        print(" ")
        print(" Error opening lower charge density file ")
        print(" ")
        quit()
    allowed_ibravs=[0,1,4,6,8,9,12,-9]
    Parameters=read_charge_parameters(chg_tot)
    if Parameters['ibrav'] in allowed_ibravs:
        Area=calc_Area(Parameters['ibrav'],Parameters['celldm'],Parameters['Lattice_Vectors'])
        Interface=get_Interface(Parameters['celldm'][0],Parameters['N_At'],Parameters['N_Ty'],chg_tot)
    else:
        print(" ")
        print(" IBRAV found in charge density file is not supported!")
        print(" Please make sure that your interface is normal to the third lattice vector.")
        print(" Use one of the following values of ibrav: "+str(allowed_ibravs))
        print(" ")
        quit()
    chg_tot.close()
    #update Parameters dictornary with number of atoms in each type
    i=0
    while i<Parameters['N_Ty']:
        Parameters['Types'][i+1].update({'Nr_of_At_in_Ty':Interface['Nr_A_in_Ty'][i+1]})
        i=i+1
    #calculate length of unit cell
    if Parameters['ibrav']==0:
        c=Parameters['Lattice_Vectors'][2][2]*Parameters['celldm'][0]*Bohr_to_Angstrom
    else:
        c=Parameters['celldm'][2]*Parameters['celldm'][0]*Bohr_to_Angstrom
    #read main data into arrays
    Data_tot=read_data(CHG_file_tot,Parameters['ibrav'],Parameters['N_At'],Parameters['N_Ty'])
    Data_upp=read_data(CHG_file_upp,Parameters['ibrav'],Parameters['N_At']/2,Parameters['N_Ty'])
    Data_lwr=read_data(CHG_file_lwr,Parameters['ibrav'],Parameters['N_At']/2,Parameters['N_Ty'])
    #Substract arrays to get charge differences
    Diff=Data_tot-Data_upp-Data_lwr
    #reshape and average data
    Line_Data=reshape_and_average(Diff,Parameters['nx'],Parameters['ny'],Parameters['nz'])
    #prepare final array with first colum the z position in Angstrom and second column the averaged charge density save this array in the file 'total_charge.dat'
    z=np.linspace(0,c,num=Parameters['nz'])
    #shift everything to the right for stuff close to zero to be rendered nicely in a figure
    n=Parameters['nz']-100
    dz=c/Parameters['nz']
    z=np.concatenate((z[n:]-max(z)-dz,z[:n]))-Interface['Int_Center']
    Line_Data=np.concatenate((Line_Data[n:],Line_Data[:n]))
    #Check value of charge in center of the interface
    check=10
    i=0
    for x in z:
        if abs(x)<check:
            check=abs(x)
            pos=i
        i=i+1
    Parameters['Center_Charge']=Line_Data[pos]
    #Construct final output array
    Final_Array=np.stack((z,Line_Data),axis=-1)
    Parameters.update(Interface)
    Parameters['Charge_Array']=Final_Array
    #Integrating P for the interface and P for the whole slab (as well as total charge which should be 0)
    Tot_Charge=intgrt.simps(Final_Array[:,1],Final_Array[:,0],even='avg')*Area
    P_total_height=intgrt.simps(np.absolute(Final_Array[:,1]),Final_Array[:,0],even='avg')/Parameters['Total_Height']
    Indexes_of_Interface=np.where(np.logical_and(z>=-Parameters['Int_Width']/2, z<=Parameters['Int_Width']/2))
    Interface_Array=np.stack((z[Indexes_of_Interface],Line_Data[Indexes_of_Interface]),axis=-1)
    P_Interface=intgrt.simps(np.absolute(Line_Data[Indexes_of_Interface]),z[Indexes_of_Interface],even='avg')/Parameters['Int_Width']
    Parameters.update({'P_Interface':P_Interface, 'P_total_height':P_total_height, 'Area':Area})
    if plotting:
        plot_charge_diff_curve(Final_Array,Parameters['Int_Width'])
    #Store final array in a separate ArrayData node The name is 'averaged_charge_array'
    out_Array=ArrayData()
    out_Array.set_array('averaged_charge_array',Final_Array)
    out_Array.store()
    #Set up final output dictionary which is somewhat smaller than the 'Parameters' dictionary
    Output_Dict={'P_Interface':Float(P_Interface), 'P_total_height':Float(P_total_height)}
    Output_Dict.update({'Charge_at_Interface_Center': Float(Parameters['Center_Charge']), 'Area':Float(Area)})
    #Output_Dict.update({'Grid_Dimensions':{'nx':Int(Parameters['nx']),'ny':Int(Parameters['ny']),'nz':Int(Parameters['nz'])}})
    Output_Dict.update({'nx':Int(Parameters['nx']),'ny':Int(Parameters['ny']),'nz':Int(Parameters['nz'])})
    Output_Dict.update({'Averaged_charge_array_PK':Int(out_Array.pk), 'Interface_Width':Float(Parameters['Int_Width'])})
    # Put output dictionary into ParameterData node
    Out_PD=ParameterData()
    Out_PD.set_dict(Output_Dict)
    Out_PD.label = u'Chargedensity difference results'
    Out_PD.description = u'Charge density difference is calculated for two slabs that form an interface. An average of the data normal to the interface is made and several characteristic numbers are computed.'
    return Out_PD
    
def plot_charge_diff_curve(Array,Width):
    x=Array[:,0]
    y=Array[:,1]*1000
    upperPlotLimit=max(y)+max(y)*0.1
    lowerPlotLimit=min(y)+min(y)*0.1
    plt.grid(False)
    px=3*Width
    plt.axis([-px, px, lowerPlotLimit, upperPlotLimit])
    plt.axvline(-Width/2,color='k',ls='--')
    plt.axvline(+Width/2,color='k',ls='--')
    plt.axhline(0,color='k')
    plt.plot(x,y,'k-', linewidth=4.0)
    plt.xlabel(r'z [$\AA$]')
    plt.ylabel(r'$\rho_\mathrm{diff}$ [me$^- \AA^{-3}$]')
    plt.savefig("RhoDiff.png", dpi=300)
    plt.gcf().clear()
    return

def reshape_and_average(Array,nx,ny,nz):
    Data=np.reshape(Array,(nz,ny,nx))*1/Bohr_to_Angstrom**3
    Average=np.mean(Data,dtype=np.float,axis=(1,2))
    return Average

def read_data(filename,ibrav,N_At,N_Ty):
    if ibrav==0:
        skip=4+N_At+N_Ty+3
    else:
        skip=4+N_At+N_Ty
    #The following fails for nx*ny*nz not devisible by 5:
    #Array=np.loadtxt(filename,dtype='float',skiprows=skip)
    #Less elegant but should work always:
    Array=[]
    with open(str(filename)) as f:
        for i in xrange(skip):
            f.next()
        for line in f:
            Array.extend(map(float,line.split()))
        Array=np.asarray(Array)
    return Array

def get_Interface(alat,N_At,N_Ty,f):
    i=0
    Nr_of_Atoms_for_each_Ty=np.zeros(N_Ty+1,dtype=int)
    zs=[]
    while i<N_At:
        nr,x,y,z,typ=f.readline().split()
        Nr_of_Atoms_for_each_Ty[int(typ)]=Nr_of_Atoms_for_each_Ty[int(typ)]+1
        i=i+1
        zs.append(float(z))
    zs=np.sort(np.asarray(zs))*Bohr_to_Angstrom
    lower=zs[N_At/2-1]*alat
    upper=zs[N_At/2]*alat
    Center=(lower+upper)/2
    Width=upper-lower
    Height=(zs[-1]-zs[0])*alat
    Interface={'Int_Center':Center,'Int_Width':Width,'Nr_A_in_Ty':Nr_of_Atoms_for_each_Ty,'Total_Height':Height}
    return Interface

def calc_Area(ibrav,celldm,LV):
    alat=celldm[0]
    if ibrav==0:
        a1=LV[0]*alat*Bohr_to_Angstrom
        a2=LV[1]*alat*Bohr_to_Angstrom
    elif ibrav==1 or ibrav==6:
        a1=np.array([1,0,0])*alat*Bohr_to_Angstrom
        a2=np.array([0,1,0])*alat*Bohr_to_Angstrom
    elif ibrav==4:
        a1=np.array([1,0,0])*alat*Bohr_to_Angstrom
        a2=np.array([-0.5,m.sqrt(3)/2,0])*alat*Bohr_to_Angstrom
    elif ibrav==8:
        ba=celldm[1]
        a1=np.array([1,0.0,0.0])*alat*Bohr_to_Angstrom
        a2=np.array([0.0,ba,0.0])*alat*Bohr_to_Angstrom
    elif ibrav==9:
        ba=celldm[1]
        a1=np.array([0.5,0.5*ba,0.0])*alat*Bohr_to_Angstrom
        a2=np.array([-0.5,0.5*ba,0.0])*alat*Bohr_to_Angstrom
    elif ibrav==-9:
        ba=celldm[1]
        a1=np.array([0.5,-0.5*ba,0.0])*alat*Bohr_to_Angstrom
        a2=np.array([0.5,0.5*ba,0.0])*alat*Bohr_to_Angstrom
    elif ibrav==12:
        ba=celldm[1]
        cos_ab=celldm[3]
        sin_ab=m.sin(m.acos(cos_ab))
        a1=np.array([1,0.0,0.0])*alat*Bohr_to_Angstrom
        a2=np.array([ba*cos_ab,ba*sin_ab,0.0])*alat*Bohr_to_Angstrom
    Area=np.linalg.norm(np.cross(a1,a2))
    return Area
        
def read_charge_parameters(f):
    #read empty line
    f.readline()
    #read first line of header and save in dictionary
    nx,ny,nz,nnx,nny,nnz,N_At,N_Ty=map(int,f.readline().split())
    P1_D={'nx':nx,'ny':ny,'nz':nz,'nnx':nnx,'nny':nny,'nnz':nnz,'N_At':N_At,'N_Ty':N_Ty}
    #read second line of header and save in dictionary
    line2=f.readline().split()
    ibrav=line2[0]
    celldm=line2[1:]
    celldm=map(float,celldm)
    P2_D={'ibrav':int(ibrav), 'celldm':celldm}
    #check if lattice vectors are set manually for ibrav=0
    if P2_D['ibrav']==0:
        #read 3 lattice vectors and save them in numpy array
        a1,a2,a3=map(float,f.readline().split())
        b1,b2,b3=map(float,f.readline().split())
        c1,c2,c3=map(float,f.readline().split())
        V=np.array([[a1,a2,a3],[b1,b2,b3],[c1,c2,c3]])
        #read line with unimportant info
        f.readline()
    else:
        V='No_explicit_lattice_vectors'
        #read line with unimportant info
        f.readline()
    # Read additional line(s) to check species and their valence
    i=0
    Types={}
    while i<N_Ty:
        index,name,valence=f.readline().split()
        Types[int(index)]={'name':name,'valence':float(valence)}
        i=i+1
    P_all=P1_D.copy()
    P_all.update(P2_D)
    P_all['Lattice_Vectors']=V
    P_all['Types']=Types
    return P_all
 
#"""
#-----------------------------------------------------------------------------------------
#Functions for PPES calculations:
#-----------------------------------------------------------------------------------------
#"""

@workfunction
def make_PPES_sequential(top_slab_structure, bottom_slab_structure, dz_min, dz_max, dz_step, codename, pseudo_family, metal, magnet, bands, vdw, kdistance, scf_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling):
        Energy_vs_dz=[]
        for dz in np.arange(float(dz_min),float(dz_max),float(dz_step)):
                new_structure = copy_structure(bottom_slab_structure)
                shifted_structure = add_shifted_upper_atoms_and_store(new_structure,top_slab_structure,dz)
                input_params = generate_scfinter_input_params(shifted_structure, codename, pseudo_family, bool(metal), bool(magnet), bands, bool(vdw), kdistance, int(scf_wallclock_max_time), energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling)
                PPES_pw_proc = PwCalculation.process()
                PPES_result = run(PPES_pw_proc, **input_params)
                PPES_result['output_parameters'].get_inputs()[0].label = "P-PES calc for dz = "+str(dz)
                PPES_result['output_parameters'].get_inputs()[0].description = "P-PES pw.x calculation for "+shifted_structure.get_formula()+" at relative displacement dz = "+str(dz)
                if PPES_result['output_parameters'].get_inputs()[0].get_state() == 'FINISHED':
                        Energy = PPES_result['output_parameters'].dict.energy
                        Energy_vs_dz.append([dz, Energy])
                else:
                        print(' PPES Calculation for dz = '+str(dz)+' failed!')
        Energy_vs_dz=np.asarray(Energy_vs_dz)
        out_Array=ArrayData()
        out_Array.set_array('PPES_array',Energy_vs_dz)
        out_Array.store()
        plot_PPES(Energy_vs_dz, Str(shifted_structure.get_formula()))
        return out_Array

@workfunction
def make_PPES_parallel(top_slab_structure, bottom_slab_structure, dz_min, dz_max, dz_step, codename, pseudo_family, metal, magnet, bands, vdw, kdistance, scf_wallclock_max_time, energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling):
        PPES_calculations={}
        for dz in np.arange(float(dz_min),float(dz_max),float(dz_step)):
                new_structure = copy_structure(bottom_slab_structure)
                shifted_structure = add_shifted_upper_atoms_and_store(new_structure,top_slab_structure,dz)
                input_params = generate_scfinter_input_params(shifted_structure, codename, pseudo_family, bool(metal), bool(magnet), bands, bool(vdw), kdistance, int(scf_wallclock_max_time), energy_cutoff_wavefunction, energy_cutoff_charge, vdW_scaling)
                PPES_pw_proc = PwCalculation.process()
                future = submit(PPES_pw_proc, **input_params)
                PPES_calculations[dz]=future
        check_if_calculations_are_done(PPES_calculations)
        out_Array = gather_PPES_energies(PPES_calculations, Str(shifted_structure.get_formula()))
        return out_Array

def gather_PPES_energies(dictionary_of_processes, System_Formula):
        Energy_vs_dz=[]
        for dz in dictionary_of_processes.keys():
                calc = load_node(dictionary_of_processes[dz].pid)
                calc.label = "P-PES calc for dz = "+str(dz)
                calc.description = "P-PES pw.x calculation for "+str(System_Formula)+" at relative displacement dz = "+str(dz)
                if calc.get_state() == 'FINISHED':
                        Energy = calc.res.energy
                        Energy_vs_dz.append([dz, Energy])
                else:
                        print(' PPES Calculation for dz = '+str(dz)+' failed!')
        Energy_vs_dz=np.asarray(sorted(Energy_vs_dz))
        out_Array=ArrayData()
        out_Array.set_array('PPES_array',Energy_vs_dz)
        out_Array.store()
        plot_PPES(Energy_vs_dz, System_Formula)
        return out_Array

def check_if_calculations_are_done(dictionary_of_processes):
        not_finished = {x: 1 for x in dictionary_of_processes}
        while any(not_finished.values()):
                time.sleep(20)
                for i in dictionary_of_processes.keys():
                        calc = load_node(dictionary_of_processes[i].pid)
                        if calc.get_state() == 'FINISHED' or calc.get_state() == 'FAILED':
                                not_finished[i] = 0
        return

def plot_PPES(array,System_Formula):
        x = array[:,0]
        y = array[:,1]-array[-1,1]
        #spline interpolation either with interp1d (free) or with CubicSpline (to use boundary conditions):
        #f = interp1d(x, y, kind='cubic')
        f = CubicSpline(x, y, bc_type=('natural','clamped'))
        x_f = np.linspace(min(x),max(x),100)
        y_f = f(x_f)
        upPlotLimit = max(y_f)+max(y_f)*0.1
        loPlotLimit = min(y_f)-0.5
        lePlotLimit = min(x)-0.25
        riPlotLimit = max(x)+0.25
        plt.axis([lePlotLimit, riPlotLimit, loPlotLimit, upPlotLimit])
        plt.plot(x_f, y_f,'k-', x,y,'ro', linewidth=2.0, markersize=8.0)
        plt.xlabel(r'$\Delta z$ [$\AA$]')
        plt.ylabel(r'Energy [eV]')
        FigName=str(System_Formula)+"_PPES.png"
        plt.savefig(FigName, dpi=400)
        plt.gcf().clear()
        return

def copy_structure(input_structure):
        StructureData = DataFactory("structure")
        the_cell = input_structure.cell
        new_structure = StructureData(cell=the_cell)
        for s in input_structure.sites:
                u1,u2,name,u3,pos = str(s).split()
                x,y,z=map(float,pos.split(','))
                new_structure.append_atom(position=(x,y,z), symbols=name.strip("'"))
        return new_structure

def add_shifted_upper_atoms_and_store(new_structure,upper_structure,dz):
        for s in upper_structure.sites:
                u1,u2,name,u3,pos = str(s).split()
                x,y,z=map(float,pos.split(','))
                new_structure.append_atom(position=(x,y,z+dz), symbols=name.strip("'"))
        new_structure.store()
        new_structure.label = "P-PES structure, dz = "+str(dz)
        new_structure.description = "P-PES structure with delta z = "+str(dz)+" Angstrom"
        return new_structure

"""
-----------------------------------------------------------------------------------------
"""

@workfunction
def converge_Encut(Starting_EncutWF, Encut_WF_increment, Encut_Chrg_Multi, tolerance, Structure_1, Structure_2, Pseudo_Fam, PW_Code_Name, is_metal, is_magnet, vdw, wallclock_max_time, vdW_scaling):
	Nr_of_Atoms = len(Structure_1.sites)
	tol = 1e10
	Encut_WF = float(Starting_EncutWF)
	Encut_CHG = Encut_WF *float( Encut_Chrg_Multi)
	Encuts_vs_Ediff=[]
	while tol > float(tolerance)*Nr_of_Atoms:
		input_params_S1 = generate_cube_input_params(Structure_1, PW_Code_Name, Pseudo_Fam, bool(is_metal), bool(is_magnet), bool(vdw), int(wallclock_max_time), Encut_WF, Encut_CHG, float(vdW_scaling))
		input_params_S2 = generate_cube_input_params(Structure_2, PW_Code_Name, Pseudo_Fam, bool(is_metal), bool(is_magnet), bool(vdw), int(wallclock_max_time), Encut_WF, Encut_CHG, float(vdW_scaling))
		Pw_proc_1 = PwCalculation.process()
		Pw_proc_2 = PwCalculation.process()
		future1 = submit(Pw_proc_1, **input_params_S1)
		future2 = submit(Pw_proc_2, **input_params_S2)
		calc1 = load_node(future1.pid)
		calc2 = load_node(future2.pid)
		calc1.label = "Cutoff Conv. Calc. for EncutWF = "+str(Encut_WF)+"Ry. (LV)"
		calc1.description = "pw.x calculation to converge the energy cutoff for "+str(Structure_1.get_formula())+". The energy cutoff for the wavefunctions is now = "+str(Encut_WF)+"Ry, and the cutoff for the charge density is "+str(Encut_CHG)+"Ry. (Low Volume)"
		calc2.label = "Cutoff Conv. Calc. for EncutWF = "+str(Encut_WF)+"Ry. (HV)"
		calc2.description = "pw.x calculation to converge the energy cutoff for "+str(Structure_1.get_formula())+". The energy cutoff for the wavefunctions is now = "+str(Encut_WF)+"Ry, and the cutoff for the charge density is "+str(Encut_CHG)+"Ry. (High Volume)"
		not_finished = [1,1]
		while any(not_finished):
			time.sleep(5)
			if calc1.get_state() == 'FINISHED' or calc1.get_state() == 'FAILED':
				not_finished[0] = 0
			if calc2.get_state() == 'FINISHED' or calc2.get_state() == 'FAILED':
				not_finished[1] = 0
		if calc1.get_state() == 'FAILED': 
                        with open('output.txt', 'a') as out:
				out.write('Critical calculation Nr.: '+str(calc1.pk)+' failed trying to converge the Energy Cutoff. Aborting Workflow!')
	    		raise SystemExit
		if calc2.get_state() == 'FAILED':
                        with open('output.txt', 'a') as out:
				out.write('Critical calculation Nr.: '+str(calc2.pk)+' failed trying to converge the Energy Cutoff. Aborting Workflow!')
	    		raise SystemExit
		E_diff=calc1.res.energy-calc2.res.energy
		Encuts_vs_Ediff.append([Encut_WF, E_diff])
		if len(Encuts_vs_Ediff) > 1:
			tol = abs(Encuts_vs_Ediff[-2][1] - Encuts_vs_Ediff[-1][1])
		Encut_WF = Encut_WF + Encut_WF_increment
		Encut_CHG = Encut_WF * Encut_Chrg_Multi
	Encuts_vs_Ediff=np.asarray(Encuts_vs_Ediff)
	Encut_Array=ArrayData()
	Encut_Array.set_array('Encut_array',Encuts_vs_Ediff)
	Encut_Array.store()
	Encut_Array.label = 'Array for Encut Conv. for '+str(Structure_1.get_formula())
	with open('output.txt', 'a') as out:
		out.write('Encut array package number = '+str(Encut_Array.pk)+'\n')
	return Float(Encuts_vs_Ediff[-2,0])


