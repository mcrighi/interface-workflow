# -*- coding: utf-8 -*-
from aiida.backends.utils import load_dbenv, is_dbenv_loaded

__license__ = "MIT license, see LICENSE.txt file."
__authors__ = "Tribology Group of the University of Modena."
__version__ = "0.9.1"

if not is_dbenv_loaded():
    load_dbenv()

import ase
import math
from aiida.orm import load_node
from aiida.orm import DataFactory
from aiida.orm.code import Code
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array import ArrayData
from aiida.work.db_types import BaseType, Str, Float, Int
#from aiida.orm.data.base import BaseType, NumericType, Str, Float, Int
from aiida.orm.data.base import NumericType
from aiida.orm.calculation.job.quantumespresso.pw import PwCalculation
from aiida.work.run import run, async
from aiida.work.workchain import (WorkChain, ToContext, while_)
from aiida.work.workfunction import workfunction
from aiida.work.defaults import registry

from numpy import array, arange

GPa_to_eV_over_ang3 = 1. / 160.21766208
eVA2_to_Jm2 = 16.0218
vacuum_slab = 12                     #Vacuum of the slab calculation for the surface energy
iter_adh = 0
iter_adh_x = 0
iter_adh_y = 0
inter_factor = 1.1
path_x = []
path_y = []
adh_x = []
adh_y = []
array_accumulo = []


def hs_dict(alat):
    
    hs_dictionary = { 'hsarray_diamond100': array([[0. * alat * (1/math.sqrt(2.)), 0. * alat * (1/math.sqrt(2.))], [0. * alat * (1/math.sqrt(2.)), 0.25 * alat * (1/math.sqrt(2.))], [0. * alat * (1/math.sqrt(2.)), 0.5 * alat * (1/math.sqrt(2.))], [0.125 * alat * (1/math.sqrt(2.)), 0.125 * alat * (1/math.sqrt(2.))], [0.125 * alat * (1/math.sqrt(2.)), 0.375 * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), 0. * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), 0.25 * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), 0.5 * alat * (1/math.sqrt(2.))], [0.375 * alat * (1/math.sqrt(2.)), 0.125 * alat * (1/math.sqrt(2.))], [0.375 * alat * (1/math.sqrt(2.)), 0.375 * alat * (1/math.sqrt(2.))], [0.5 * alat * (1/math.sqrt(2.)), 0. * alat * (1/math.sqrt(2.))], [0.5 * alat * (1/math.sqrt(2.)), 0.25 * alat * (1/math.sqrt(2.))], [0.5 * alat * (1/math.sqrt(2.)), 0.5 * alat * (1/math.sqrt(2.))]]),
		      'hsarray_diamond110': array([[0. * alat * (1/math.sqrt(2.)), 0. * alat * (1/math.sqrt(2.))], [0. * alat * (1/math.sqrt(2.)), (math.sqrt(2.)/4.) * alat * (1/math.sqrt(2.))], [0. * alat * (1/math.sqrt(2.)), (math.sqrt(2.)/2.) * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), 0. * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), (math.sqrt(2.)/4.) * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), (math.sqrt(2.)/2.) * alat * (1/math.sqrt(2.))]]),
		      'hsarray_diamond111': array([[0. * alat * (1/math.sqrt(2.)), 0. * alat * (1/math.sqrt(2.))], [0. * alat * (1/math.sqrt(2.)), (math.sqrt(3.)/6.) * alat * (1/math.sqrt(2.))], [0. * alat * (1/math.sqrt(2.)), (math.sqrt(3.)/3.) * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), (math.sqrt(3.)/12.) * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), (math.sqrt(3.)/4.) * alat * (1/math.sqrt(2.))], [0.5 * alat * (1/math.sqrt(2.)), (math.sqrt(3.)/6.) * alat * (1/math.sqrt(2.))]]),
		      'hsarray_fcc100': array([[0. * alat * (1/math.sqrt(2.)), 0. * alat * (1/math.sqrt(2.))], [0. * alat * (1/math.sqrt(2.)), 0.25 * alat * (1/math.sqrt(2.))], [0. * alat * (1/math.sqrt(2.)), 0.5 * alat * (1/math.sqrt(2.))], [0.125 * alat * (1/math.sqrt(2.)), 0.125 * alat * (1/math.sqrt(2.))], [0.125 * alat * (1/math.sqrt(2.)), 0.375 * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), 0.25 * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), 0.5 * alat * (1/math.sqrt(2.))], [0.375 * alat * (1/math.sqrt(2.)), 0.375 * alat * (1/math.sqrt(2.))], [0.5 * alat * (1/math.sqrt(2.)), 0.5 * alat * (1/math.sqrt(2.))]]),
		      'hsarray_fcc110': array([[0. * alat * (1/math.sqrt(2.)), 0. * alat * (1/math.sqrt(2.))], [0. * alat * (1/math.sqrt(2.)), (math.sqrt(2.)/4.) * alat * (1/math.sqrt(2.))], [0. * alat * (1/math.sqrt(2.)), (math.sqrt(2.)/2.) * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), 0. * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), (math.sqrt(2.)/4.) * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), (math.sqrt(2.)/2.) * alat * (1/math.sqrt(2.))], [0.5 * alat * (1/math.sqrt(2.)), 0. * alat * (1/math.sqrt(2.))], [0.5 * alat * (1/math.sqrt(2.)), (math.sqrt(2.)/4.) * alat * (1/math.sqrt(2.))], [0.5 * alat * (1/math.sqrt(2.)), (math.sqrt(2.)/2.) * alat * (1/math.sqrt(2.))]]),
		      'hsarray_fcc111': array([[0. * alat * (1/math.sqrt(2.)), 0. * alat * (1/math.sqrt(2.))], [0. * alat * (1/math.sqrt(2.)), (math.sqrt(3.)/6.) * alat * (1/math.sqrt(2.))], [0. * alat * (1/math.sqrt(2.)), (math.sqrt(3.)/3.) * alat * (1/math.sqrt(2.))],[0.25 * alat * (1/math.sqrt(2.)), (math.sqrt(3.)/12.) * alat * (1/math.sqrt(2.))], [0.25 * alat * (1/math.sqrt(2.)), (math.sqrt(3.)/4.) * alat * (1/math.sqrt(2.))], [0.5 * alat * (1/math.sqrt(2.)), (math.sqrt(3.)/6.) * alat * (1/math.sqrt(2.))]]),
		      'hsarray_bcc100': array([[0. * alat, 0. * alat], [0. * alat, 0.25 * alat], [0. * alat, 0.5 * alat], [0.125 * alat, 0.125 * alat], [0.125 * alat, 0.375 * alat], [0.25 * alat, 0.25 * alat], [0.25 * alat, 0.5 * alat], [0.375 * alat, 0.375 * alat], [0.5 * alat, 0.5 * alat]]),
		      'hsarray_bcc110': array([[0. * alat, 0. * alat * (math.sqrt(2.))], [0.25 * alat, 0.0 * alat * (math.sqrt(2.))], [0.25 * alat, 0.25 * alat * (math.sqrt(2.))], [0.5 * alat, 0. * alat * (math.sqrt(2.))], [0.5 * alat, 0.25 * alat * (math.sqrt(2.))]]),
		      'hsarray_bcc111': array([[0. * alat * (math.sqrt(2.)), 0. * alat * (math.sqrt(2.))], [0. * alat * (math.sqrt(2.)), (math.sqrt(3.)/6.) * alat * (math.sqrt(2.))], [0. * alat * (math.sqrt(2.)), (math.sqrt(3.)/3.) * alat * (math.sqrt(2.))], [0.25 * alat * (math.sqrt(2.)), (math.sqrt(3.)/12.) * alat * (math.sqrt(2.))], [0.25 * alat * (math.sqrt(2.)), (math.sqrt(3.)/4.) * alat * (math.sqrt(2.))], [0.5 * alat * (math.sqrt(2.)), (math.sqrt(3.)/6.) * alat * (math.sqrt(2.))]]),
		      'hsarray_graphene111': array([[0. * alat, 0. * alat], [0. * alat, (math.sqrt(3.)/6.) * alat], [0. * alat, (math.sqrt(3.)/3.) * alat],[0.25 * alat, (math.sqrt(3.)/12.) * alat], [0.25 * alat, (math.sqrt(3.)/4.) * alat], [0.5 * alat, (math.sqrt(3.)/6.) * alat]]),
		      'hsarray_hcp0001': array([[0. * alat, 0. * alat], [0. * alat, (math.sqrt(3.)/6.) * alat], [0. * alat, (math.sqrt(3.)/3.) * alat],[0.25 * alat, (math.sqrt(3.)/12.) * alat], [0.25 * alat, (math.sqrt(3.)/4.) * alat], [0.5 * alat, (math.sqrt(3.)/6.) * alat]])
		    }

    return hs_dictionary



@workfunction
def diamond(element,alat):
    """
    Workfunction to create a diamond crystal structure with a given element.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    the_cell = array([[0., 0.5, 0.5],
                [0.5, 0., 0.5],
                [0.5, 0.5, 0.]]) * alat
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(0., 0., 0.), symbols=str(element))
    structure.append_atom(position=(0.25 * alat, 0.25 * alat, 0.25 * alat), symbols=str(element))

    return structure


@workfunction
def fcc(element,alat):
    """
    Workfunction to create a fcc crystal structure with a given element.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    the_cell = array([[0., 0.5, 0.5],
                [0.5, 0., 0.5],
                [0.5, 0.5, 0.]]) * alat
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(0., 0., 0.), symbols=str(element))

    return structure


@workfunction
def bcc(element,alat):
    """
    Workfunction to create a bcc crystal structure with a given element.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    the_cell = array([[0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, -0.5, 0.5]]) * alat
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(0., 0., 0.), symbols=str(element))

    return structure


@workfunction
def graphene(element,alat):
    """
    Workfunction to create a graphene-like crystal structure with a given element.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    the_cell = array([[1.0, 0., 0.],
                [-0.5, math.sqrt(3) / 2., 0.],
                [0., 0., 0.]]) * alat
    the_cell[2][2] += 3.33                                                          
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(0., 0., 0.), symbols=str(element))
    structure.append_atom(position=(0.5 * alat, (math.sqrt(3.)/6.) * alat, 0. * alat), symbols=str(element))

    return structure


@workfunction
def hcp(element,alat):
    """
    Workfunction to create a hcp crystal structure with a given element.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    the_cell = array([[1.0, 0., 0.],
                [-0.5, math.sqrt(3) / 2., 0.],
                [0., 0., 2 * math.sqrt(2. / 3.)]]) * alat
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(0., 0., 0.), symbols=str(element))
    structure.append_atom(position=(0.5 * alat, (math.sqrt(3.)/6.) * alat, math.sqrt(2. / 3.) * alat), symbols=str(element))

    return structure


@workfunction
def diamond100_bulk(element,alat):
    """
    Workfunction to create a bulk diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * (1./math.sqrt(2.))
    the_cell = array([[1.0, 0., 0.], [0.0, 1.0 , 0.0], [0, 0, math.sqrt(2.)]]) * alat_cell
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * 0.0, alat_cell * (0 * math.sqrt(2.)/4.)), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * 0.0, alat_cell * (1 * math.sqrt(2.)/4.)), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * 0.5, alat_cell * (2 * math.sqrt(2.)/4.)), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * 0.5, alat_cell * (3 * math.sqrt(2.)/4.)), symbols=str(element))

    return structure


@workfunction
def diamond100_slab(element, alat):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * (1./math.sqrt(2.))
    cell_unit = array([[1.0, 0., 0.], [0.0, 1.0 , 0.0], [0, 0, math.sqrt(2.)]]) * alat_cell                  #unit cell
    rip_vet = array([1, 1, 3])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                          #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * 0.0 + dy, alat_cell * (0 * math.sqrt(2.)/4.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * 0.0 + dy, alat_cell * (1 * math.sqrt(2.)/4.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * 0.5 + dy, alat_cell * (2 * math.sqrt(2.)/4.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * 0.5 + dy, alat_cell * (3 * math.sqrt(2.)/4.) + dz), symbols=str(element))

    return structure


@workfunction
def diamond100_inter(element, alat, hs_vect):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    hs_np_vect = hs_vect.get_array('hs_array')
    alat_cell = alat * (1./math.sqrt(2.))
    cell_unit = array([[1.0, 0., 0.], [0.0, 1.0 , 0.0], [0, 0, math.sqrt(2.)]]) * alat_cell                  #unit cell
    rip_vet = array([1, 1, 3])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                          #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0], alat_cell * 0.0 + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(4., -1) + alat_cell * (0 * math.sqrt(2.)/4.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx + hs_np_vect[0], alat_cell * 0.0 + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(4., -1) + alat_cell * (1 * math.sqrt(2.)/4.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx + hs_np_vect[0], alat_cell * 0.5 + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(4., -1) + alat_cell * (2 * math.sqrt(2.)/4.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0], alat_cell * 0.5 + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(4., -1) + alat_cell * (3 * math.sqrt(2.)/4.) + dz), symbols=str(element))
		structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * 0.0 + dy, -1 * ( alat_cell * (0 * math.sqrt(2.)/4.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * 0.0 + dy, -1 * ( alat_cell * (1 * math.sqrt(2.)/4.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * 0.5 + dy, -1 * ( alat_cell * (2 * math.sqrt(2.)/4.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * 0.5 + dy, -1 * ( alat_cell * (3 * math.sqrt(2.)/4.) + dz) ), symbols=str(element))

    return structure


@workfunction
def diamond110_bulk(element,alat):
    """
    Workfunction to create a bulk diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * (1./math.sqrt(2.))
    the_cell = array([[1.0, 0., 0.], [0., math.sqrt(2.) , 0.], [0., 0., 1.0]]) * alat_cell
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * (0 * math.sqrt(2.)/4.), alat_cell * 0.0), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * (1 * math.sqrt(2.)/4.), alat_cell * 0.0), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * (2 * math.sqrt(2.)/4.), alat_cell * 0.5), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * (3 * math.sqrt(2.)/4.), alat_cell * 0.5), symbols=str(element))

    return structure


@workfunction
def diamond110_slab(element, alat):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * (1./math.sqrt(2.))
    cell_unit = array([[1.0, 0., 0.], [0.0, math.sqrt(2.) , 0.0], [0, 0, 1.0]]) * alat_cell                  #unit cell
    rip_vet = array([1, 1, 3])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * (0 * math.sqrt(2.)/4.) + dy, alat_cell * 0.0 + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * (1 * math.sqrt(2.)/4.) + dy, alat_cell * 0.0 + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * (2 * math.sqrt(2.)/4.) + dy, alat_cell * 0.5 + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * (3 * math.sqrt(2.)/4.) + dy, alat_cell * 0.5 + dz), symbols=str(element))

    return structure


@workfunction
def diamond110_inter(element, alat, hs_vect):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    hs_np_vect = hs_vect.get_array('hs_array')
    alat_cell = alat * (1./math.sqrt(2.))
    cell_unit = array([[1.0, 0., 0.], [0.0, math.sqrt(2.) , 0.0], [0, 0, 1.0]]) * alat_cell                  #unit cell
    rip_vet = array([1, 1, 3])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0], alat_cell * (0 * math.sqrt(2.)/4.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(4., -1) + alat_cell * 0.0 + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx + hs_np_vect[0], alat_cell * (1 * math.sqrt(2.)/4.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(4., -1) + alat_cell * 0.0 + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx + hs_np_vect[0], alat_cell * (2 * math.sqrt(2.)/4.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(4., -1) + alat_cell * 0.5 + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0], alat_cell * (3 * math.sqrt(2.)/4.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(4., -1) + alat_cell * 0.5 + dz), symbols=str(element))
		structure.append_atom(position=(alat_cell * 0.0 + dx, -1 * (alat_cell * (0 * math.sqrt(2.)/4.) + dy), -1 * ( alat_cell * 0.0 + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, -1 * (alat_cell * (1 * math.sqrt(2.)/4.) + dy), -1 * ( alat_cell * 0.0 + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, -1 * (alat_cell * (2 * math.sqrt(2.)/4.) + dy), -1 * ( alat_cell * 0.5 + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx, -1 * (alat_cell * (3 * math.sqrt(2.)/4.) + dy), -1 * ( alat_cell * 0.5 + dz) ), symbols=str(element))

    return structure


@workfunction
def diamond111_bulk(element,alat):
    """
    Workfunction to create a bulk diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * (1./math.sqrt(2.))
    the_cell = array([[1.0, 0., 0.], [- 0.5, math.sqrt(3) / 2., 0], [0, 0, math.sqrt(6.)]]) * alat_cell
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * (0 * math.sqrt(3.)/6.), alat_cell * (0 * math.sqrt(6.)/12.)), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * (1 * math.sqrt(3.)/6.), alat_cell * (1 * math.sqrt(6.)/12.)), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * (1 * math.sqrt(3.)/6.), alat_cell * (4 * math.sqrt(6.)/12.)), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * (2 * math.sqrt(3.)/6.), alat_cell * (5 * math.sqrt(6.)/12.)), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * (2 * math.sqrt(3.)/6.), alat_cell * (8 * math.sqrt(6.)/12.)), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * (0 * math.sqrt(3.)/6.), alat_cell * (9 * math.sqrt(6.)/12.)), symbols=str(element))

    return structure


@workfunction
def diamond111_slab(element, alat):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * (1./math.sqrt(2.))
    cell_unit = array([[1.0, 0., 0.], [- 0.5, math.sqrt(3) / 2., 0], [0, 0, math.sqrt(6.)]]) * alat_cell     #unit cell
    rip_vet = array([1, 1, 2])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx,  alat_cell * (0 * math.sqrt(3.)/6.) + dy,  alat_cell * (0 * math.sqrt(6.)/12.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx,  alat_cell * (1 * math.sqrt(3.)/6.) + dy,  alat_cell * (1 * math.sqrt(6.)/12.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx,  alat_cell * (1 * math.sqrt(3.)/6.) + dy,  alat_cell * (4 * math.sqrt(6.)/12.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx,  alat_cell * (2 * math.sqrt(3.)/6.) + dy,  alat_cell * (5 * math.sqrt(6.)/12.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx,  alat_cell * (2 * math.sqrt(3.)/6.) + dy,  alat_cell * (8 * math.sqrt(6.)/12.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx,  alat_cell * (0 * math.sqrt(3.)/6.) + dy,  alat_cell * (9 * math.sqrt(6.)/12.) + dz), symbols=str(element))

    return structure


@workfunction
def diamond111_inter(element, alat, hs_vect):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    hs_np_vect = hs_vect.get_array('hs_array')
    alat_cell = alat * (1./math.sqrt(2.))
    cell_unit = array([[1.0, 0., 0.], [- 0.5, math.sqrt(3) / 2., 0], [0, 0, math.sqrt(6.)]]) * alat_cell     #unit cell
    rip_vet = array([1, 1, 2])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0],  alat_cell * (0 * math.sqrt(3.)/6.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(3., -1) + alat_cell * (0 * math.sqrt(6.)/12.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx + hs_np_vect[0],  alat_cell * (1 * math.sqrt(3.)/6.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(3., -1) + alat_cell * (1 * math.sqrt(6.)/12.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx + hs_np_vect[0],  alat_cell * (1 * math.sqrt(3.)/6.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(3., -1) + alat_cell * (4 * math.sqrt(6.)/12.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0],  alat_cell * (2 * math.sqrt(3.)/6.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(3., -1) + alat_cell * (5 * math.sqrt(6.)/12.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0],  alat_cell * (2 * math.sqrt(3.)/6.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(3., -1) + alat_cell * (8 * math.sqrt(6.)/12.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0],  alat_cell * (0 * math.sqrt(3.)/6.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(3., -1) + alat_cell * (9 * math.sqrt(6.)/12.) + dz), symbols=str(element))
		structure.append_atom(position=(alat_cell * 0.0 + dx, -1 * ( alat_cell * (0 * math.sqrt(3.)/6.) + dy), -1 * ( alat_cell * (0 * math.sqrt(6.)/12.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, -1 * ( alat_cell * (1 * math.sqrt(3.)/6.) + dy), -1 * ( alat_cell * (1 * math.sqrt(6.)/12.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, -1 * ( alat_cell * (1 * math.sqrt(3.)/6.) + dy), -1 * ( alat_cell * (4 * math.sqrt(6.)/12.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx, -1 * ( alat_cell * (2 * math.sqrt(3.)/6.) + dy), -1 * ( alat_cell * (5 * math.sqrt(6.)/12.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx, -1 * ( alat_cell * (2 * math.sqrt(3.)/6.) + dy), -1 * ( alat_cell * (8 * math.sqrt(6.)/12.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx, -1 * ( alat_cell * (0 * math.sqrt(3.)/6.) + dy), -1 * ( alat_cell * (9 * math.sqrt(6.)/12.) + dz) ), symbols=str(element))

    return structure


@workfunction
def fcc100_bulk(element,alat):
    """
    Workfunction to create a bulk diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * (1./math.sqrt(2.)) 
    the_cell = array([[1.0, 0., 0.], [0.0, 1.0 , 0.0], [0, 0, math.sqrt(2.)]]) * alat_cell
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * 0.0, alat_cell * 0.0 * math.sqrt(2.)), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * 0.5, alat_cell * 0.5 * math.sqrt(2.)), symbols=str(element))

    return structure


@workfunction
def fcc100_slab(element, alat):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * (1./math.sqrt(2.))
    cell_unit = array([[1.0, 0., 0.], [0.0, 1.0 , 0.0], [0, 0, math.sqrt(2.)]]) * alat_cell                  #unit cell
    rip_vet = array([1, 1, 3])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * 0.0 + dy, alat_cell * 0.0 * math.sqrt(2.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * 0.5 + dy, alat_cell * 0.5 * math.sqrt(2.) + dz), symbols=str(element))

    return structure


@workfunction
def fcc100_inter(element, alat, hs_vect):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    hs_np_vect = hs_vect.get_array('hs_array')
    alat_cell = alat * (1./math.sqrt(2.))
    cell_unit = array([[1.0, 0., 0.], [0.0, 1.0 , 0.0], [0, 0, math.sqrt(2.)]]) * alat_cell                  #unit cell
    rip_vet = array([1, 1, 3])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
                structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0], alat_cell * 0.0 + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(2., -1) + alat_cell * 0.0 * math.sqrt(2.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx + hs_np_vect[0], alat_cell * 0.5 + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(2., -1) + alat_cell * 0.5 * math.sqrt(2.) + dz), symbols=str(element))
	        structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * 0.0 + dy, -1 * (alat_cell * 0.0 * math.sqrt(2.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * 0.5 + dy, -1 * (alat_cell * 0.5 * math.sqrt(2.) + dz) ), symbols=str(element))

    return structure


@workfunction
def fcc110_bulk(element,alat):
    """
    Workfunction to create a bulk diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * (1./math.sqrt(2.))
    the_cell = array([[1.0, 0., 0.], [0., math.sqrt(2.) , 0.], [0., 0., 1.0]]) * alat_cell
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * 0.0 * math.sqrt(2.), alat_cell * 0.0), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * 0.5 * math.sqrt(2.), alat_cell * 0.5), symbols=str(element))

    return structure


@workfunction
def fcc110_slab(element, alat):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * (1./math.sqrt(2.))
    cell_unit = array([[1.0, 0., 0.], [0.0, math.sqrt(2.) , 0.0], [0, 0, 1.0]]) * alat_cell                  #unit cell
    rip_vet = array([1, 1, 3])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * 0.0 * math.sqrt(2.) + dy, alat_cell * 0.0 + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * 0.5 * math.sqrt(2.) + dy, alat_cell * 0.5 + dz), symbols=str(element))

    return structure


@workfunction
def fcc110_inter(element, alat, hs_vect):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    hs_np_vect = hs_vect.get_array('hs_array')
    alat_cell = alat * (1./math.sqrt(2.))
    cell_unit = array([[1.0, 0., 0.], [0.0, math.sqrt(2.) , 0.0], [0, 0, 1.0]]) * alat_cell                  #unit cell
    rip_vet = array([1, 1, 3])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0], alat_cell * 0.0 * math.sqrt(2.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(2., -1) + alat_cell * 0.0 + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx + hs_np_vect[0], alat_cell * 0.5 * math.sqrt(2.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(2., -1) + alat_cell * 0.5 + dz), symbols=str(element))
		structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * 0.0 * math.sqrt(2.) + dy, -1 * ( alat_cell * 0.0 + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * 0.5 * math.sqrt(2.) + dy, -1 * ( alat_cell * 0.5 + dz) ), symbols=str(element))

    return structure


@workfunction
def fcc111_bulk(element,alat):
    """
    Workfunction to create a bulk diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * (1./math.sqrt(2.))
    the_cell = array([[1.0, 0., 0.], [- 0.5, math.sqrt(3) / 2., 0], [0, 0, math.sqrt(6.)]]) * alat_cell
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * (0 * math.sqrt(3.)/6.), alat_cell * (0 * math.sqrt(6.)/3.)), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * (1 * math.sqrt(3.)/6.), alat_cell * (1 * math.sqrt(6.)/3.)), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * (2 * math.sqrt(3.)/6.), alat_cell * (2 * math.sqrt(6.)/3.)), symbols=str(element))

    return structure


@workfunction
def fcc111_slab(element, alat):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * (1./math.sqrt(2.))
    cell_unit = array([[1.0, 0., 0.], [- 0.5, math.sqrt(3) / 2., 0], [0, 0, math.sqrt(6.)]]) * alat_cell     #unit cell
    rip_vet = array([1, 1, 2])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx,  alat_cell * (0 * math.sqrt(3.)/6.) + dy,  alat_cell * (0 * math.sqrt(6.)/3.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx,  alat_cell * (1 * math.sqrt(3.)/6.) + dy,  alat_cell * (1 * math.sqrt(6.)/3.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx,  alat_cell * (2 * math.sqrt(3.)/6.) + dy,  alat_cell * (2 * math.sqrt(6.)/3.) + dz), symbols=str(element))

    return structure


@workfunction
def fcc111_inter(element, alat, hs_vect):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    hs_np_vect = hs_vect.get_array('hs_array')
    alat_cell = alat * (1./math.sqrt(2.))
    cell_unit = array([[1.0, 0., 0.], [- 0.5, math.sqrt(3) / 2., 0], [0, 0, math.sqrt(6.)]]) * alat_cell     #unit cell
    rip_vet = array([1, 1, 2])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0],  alat_cell * (0 * math.sqrt(3.)/6.) + dy + hs_np_vect[1],  inter_factor*cell_unit[2][2] * math.pow(3., -1) + alat_cell * (0 * math.sqrt(6.)/3.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx + hs_np_vect[0],  alat_cell * (1 * math.sqrt(3.)/6.) + dy + hs_np_vect[1],  inter_factor*cell_unit[2][2] * math.pow(3., -1) + alat_cell * (1 * math.sqrt(6.)/3.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0],  alat_cell * (2 * math.sqrt(3.)/6.) + dy + hs_np_vect[1],  inter_factor*cell_unit[2][2] * math.pow(3., -1) + alat_cell * (2 * math.sqrt(6.)/3.) + dz), symbols=str(element))
	        structure.append_atom(position=(alat_cell * 0.0 + dx, -1 * ( alat_cell * (0 * math.sqrt(3.)/6.) + dy), -1 * ( alat_cell * (0 * math.sqrt(6.)/3.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, -1 * ( alat_cell * (1 * math.sqrt(3.)/6.) + dy), -1 * ( alat_cell * (1 * math.sqrt(6.)/3.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.0 + dx, -1 * ( alat_cell * (2 * math.sqrt(3.)/6.) + dy), -1 * ( alat_cell * (2 * math.sqrt(6.)/3.) + dz) ), symbols=str(element))

    return structure


@workfunction
def bcc100_bulk(element,alat):
    """
    Workfunction to create a bulk diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat
    the_cell = array([[1.0, 0., 0.], [0.0, 1.0, 0.0], [0, 0, 1.0]]) * alat_cell
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * 0.0, alat_cell * 0.0), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * 0.5, alat_cell * 0.5), symbols=str(element))

    return structure


@workfunction
def bcc100_slab(element, alat):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat
    cell_unit = array([[1.0, 0., 0.], [0.0, 1.0, 0.0], [0, 0, 1.0]]) * alat_cell                             #unit cell
    rip_vet = array([1, 1, 3])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * 0.0 + dy, alat_cell * 0.0 + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * 0.5 + dy, alat_cell * 0.5 + dz), symbols=str(element))

    return structure


@workfunction
def bcc100_inter(element, alat, hs_vect):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    hs_np_vect = hs_vect.get_array('hs_array')
    alat_cell = alat
    cell_unit = array([[1.0, 0., 0.], [0.0, 1.0, 0.0], [0, 0, 1.0]]) * alat_cell                             #unit cell
    rip_vet = array([1, 1, 3])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0], alat_cell * 0.0 + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(2., -1) + alat_cell * 0.0 + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx + hs_np_vect[0], alat_cell * 0.5 + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(2., -1) + alat_cell * 0.5 + dz), symbols=str(element))
	        structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * 0.0 + dy, -1 * ( alat_cell * 0.0 + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * 0.5 + dy, -1 * ( alat_cell * 0.5 + dz) ), symbols=str(element))


    return structure


@workfunction
def bcc110_bulk(element,alat):
    """
    Workfunction to create a bulk diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat
    the_cell = array([[0.5, -math.sqrt(2.)/2., 0.], [0.5, math.sqrt(2.)/2., 0.0], [0, 0, math.sqrt(2.)]]) * alat_cell
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * 0.0, alat_cell * (0.0 * math.sqrt(2.))), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * 0.0, alat_cell * (0.5 * math.sqrt(2.))), symbols=str(element))

    return structure


@workfunction
def bcc110_slab(element, alat):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat
    cell_unit = array([[0.5, -math.sqrt(2.)/2., 0.], [0.5, math.sqrt(2.)/2., 0.0], [0, 0, math.sqrt(2.)]]) * alat_cell
    rip_vet = array([1, 1, 3])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
	        structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * 0.0 + dy, alat_cell * (0.0 * math.sqrt(2.)) + dz), symbols=str(element))
    		structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * 0.0 + dy, alat_cell * (0.5 * math.sqrt(2.)) + dz), symbols=str(element))

    return structure


@workfunction
def bcc110_inter(element, alat, hs_vect):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    hs_np_vect = hs_vect.get_array('hs_array')
    alat_cell = alat
    cell_unit = array([[0.5, -math.sqrt(2.)/2., 0.], [0.5, math.sqrt(2.)/2., 0.0], [0, 0, math.sqrt(2.)]]) * alat_cell
    rip_vet = array([1, 1, 3])                                                                               #unit of ripetition
    the_cell = cell_unit*rip_vet                                                                             #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                        #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
		structure.append_atom(position=(alat_cell * 0.0 + dx + hs_np_vect[0], alat_cell * 0.0 + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(2., -1) + alat_cell * (0.0 * math.sqrt(2.)) + dz), symbols=str(element))
    		structure.append_atom(position=(alat_cell * 0.5 + dx + hs_np_vect[0], alat_cell * 0.0 + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(2., -1) + alat_cell * (0.5 * math.sqrt(2.)) + dz), symbols=str(element))
		structure.append_atom(position=(alat_cell * 0.0 + dx, alat_cell * 0.0 + dy, -1 * ( alat_cell * (0.0 * math.sqrt(2.)) + dz) ), symbols=str(element))
    		structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * 0.0 + dy, -1 * ( alat_cell * (0.5 * math.sqrt(2.)) + dz) ), symbols=str(element))

    return structure


@workfunction
def bcc111_bulk(element,alat):
    """
    Workfunction to create a bulk diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * math.sqrt(2.)
    the_cell = array([[1.0, 0., 0.], [-0.5, math.sqrt(3.) / 2., 0.0], [0, 0, math.sqrt(6.) / 4.]]) * alat_cell
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(alat * 0.0 * math.sqrt(2.), alat * (0 * math.sqrt(6.)/6.), alat * (0 * math.sqrt(3.)/6.)), symbols=str(element))
    structure.append_atom(position=(alat * 0.5 * math.sqrt(2.), alat * (1 * math.sqrt(6.)/6.), alat * (1 * math.sqrt(3.)/6.)), symbols=str(element))
    structure.append_atom(position=(alat * 0.0 * math.sqrt(2.), alat * (2 * math.sqrt(6.)/6.), alat * (2 * math.sqrt(3.)/6.)), symbols=str(element))

    return structure


@workfunction
def bcc111_slab(element, alat):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat * math.sqrt(2.)
    cell_unit = array([[1.0, 0., 0.], [-0.5, math.sqrt(3.) / 2., 0.0], [0, 0, math.sqrt(6.) / 4.]]) * alat_cell    #unit cell
    rip_vet = array([1, 1, 2])                                                                                     #unit of repetition
    the_cell = cell_unit*rip_vet                                                                                   #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                              #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
                structure.append_atom(position=(alat * 0.0 * math.sqrt(2.) + dx, alat * (0 * math.sqrt(6.)/6.) + dy, alat * (0 * math.sqrt(3.)/6.) + dz), symbols=str(element))
                structure.append_atom(position=(alat * 0.5 * math.sqrt(2.) + dx, alat * (1 * math.sqrt(6.)/6.) + dy, alat * (1 * math.sqrt(3.)/6.) + dz), symbols=str(element))
                structure.append_atom(position=(alat * 0.0 * math.sqrt(2.) + dx, alat * (2 * math.sqrt(6.)/6.) + dy, alat * (2 * math.sqrt(3.)/6.) + dz), symbols=str(element))

    return structure


@workfunction
def bcc111_inter(element, alat, hs_vect):
    """
    Workfunction to create a slab diamond crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    hs_np_vect = hs_vect.get_array('hs_array')
    alat_cell = alat * math.sqrt(2.)
    cell_unit = array([[1.0, 0., 0.], [-0.5, math.sqrt(3.) / 2., 0.0], [0, 0, math.sqrt(6.) / 4.]]) * alat_cell    #unit cell
    rip_vet = array([1, 1, 2])                                                                                     #unit of repetition
    the_cell = cell_unit*rip_vet                                                                                   #supercell for QE calculation
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab                                                              #create vacuum along z
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
	        dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
	        dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
	        dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
                structure.append_atom(position=(alat * 0.0 * math.sqrt(2.) + dx + hs_np_vect[0], alat * (0 * math.sqrt(6.)/6.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(3., -1) + alat * (0 * math.sqrt(3.)/6.) + dz), symbols=str(element))
                structure.append_atom(position=(alat * 0.5 * math.sqrt(2.) + dx + hs_np_vect[0], alat * (1 * math.sqrt(6.)/6.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(3., -1) + alat * (1 * math.sqrt(3.)/6.) + dz), symbols=str(element))
                structure.append_atom(position=(alat * 0.0 * math.sqrt(2.) + dx + hs_np_vect[0], alat * (2 * math.sqrt(6.)/6.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(3., -1) + alat * (2 * math.sqrt(3.)/6.) + dz), symbols=str(element))
		structure.append_atom(position=(alat * 0.0 * math.sqrt(2.) + dx, -1 * (alat * (0 * math.sqrt(6.)/6.) + dy), -1 * ( alat * (0 * math.sqrt(3.)/6.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat * 0.5 * math.sqrt(2.) + dx, -1 * (alat * (1 * math.sqrt(6.)/6.) + dy), -1 * ( alat * (1 * math.sqrt(3.)/6.) + dz) ), symbols=str(element))
                structure.append_atom(position=(alat * 0.0 * math.sqrt(2.) + dx, -1 * (alat * (2 * math.sqrt(6.)/6.) + dy), -1 * ( alat * (2 * math.sqrt(3.)/6.) + dz) ), symbols=str(element))

    return structure


@workfunction
def graphene111_bulk(element,alat):
    """
    Workfunction to create a bulk graphene-like crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat
    the_cell = array([[1.0, 0., 0.], [-0.5, math.sqrt(3.) / 2., 0.0], [0., 0., 0.]]) * alat_cell
    the_cell[2][2] += 6.66                                                          
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * (0 * math.sqrt(3.)/6.), alat_cell * 0 ), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * (1 * math.sqrt(3.)/6.), alat_cell * 0 ), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * (1 * math.sqrt(3.)/6.), 3.33 ), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * (2 * math.sqrt(3.)/6.), 3.33 ), symbols=str(element))

    return structure


@workfunction
def graphene111_slab(element,alat):
    """
    Workfunction to create a surface graphene-like crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat
    the_cell = array([[1.0, 0., 0.], [-0.5, math.sqrt(3.) / 2., 0.0], [0., 0., 0.]]) * alat_cell
    the_cell[2][2] += vacuum_slab + 4.0                                                          
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * (0 * math.sqrt(3.)/6.), alat_cell * 0 ), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * (1 * math.sqrt(3.)/6.), alat_cell * 0 ), symbols=str(element))

    return structure


@workfunction
def graphene111_inter(element, alat, hs_vect):
    """
    Workfunction to create a surface graphene-like crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    hs_np_vect = hs_vect.get_array('hs_array')
    alat_cell = alat
    the_cell = array([[1.0, 0., 0.], [-0.5, math.sqrt(3.) / 2., 0.0], [0., 0., 0.]]) * alat_cell
    the_cell[2][2] += vacuum_slab + 4.0                                                        
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(alat_cell * 0.0 + hs_np_vect[0], alat_cell * (0 * math.sqrt(3.)/6.) + hs_np_vect[1], alat_cell * 0 + 3.0 ), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5 + hs_np_vect[0], alat_cell * (1 * math.sqrt(3.)/6.) + hs_np_vect[1], alat_cell * 0 + 3.0 ), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.0, alat_cell * (0 * math.sqrt(3.)/6.), alat_cell * 0 ), symbols=str(element))
    structure.append_atom(position=(alat_cell * 0.5, alat_cell * (1 * math.sqrt(3.)/6.), alat_cell * 0 ), symbols=str(element))

    return structure


@workfunction
def hcp0001_bulk(element,alat):
    """
    Workfunction to create a hcp(0001) crystal structure with a given element.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    the_cell = array([[1.0, 0., 0.],
                [-0.5, math.sqrt(3) / 2., 0.],
                [0., 0., 2 * math.sqrt(2. / 3.)]]) * alat
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    structure.append_atom(position=(0., 0., 0.), symbols=str(element))
    structure.append_atom(position=(0.5 * alat, (math.sqrt(3.)/6.) * alat, math.sqrt(2. / 3.) * alat), symbols=str(element))

    return structure


@workfunction
def hcp0001_slab(element,alat):
    """
    Workfunction to create a hcp(0001) crystal structure with a given element for the calculation of the surface energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    alat_cell = alat
    cell_unit = array([[1.0, 0., 0.], [-0.5, math.sqrt(3) / 2., 0.], [0., 0., 2 * math.sqrt(2. / 3.)]]) * alat_cell
    rip_vet = array([1, 1, 3])
    the_cell = cell_unit*rip_vet                     
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
                dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
                dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
                dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
                structure.append_atom(position=(alat_cell * 0. + dx, alat_cell * 0. + dy, alat_cell * 0. + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, alat_cell * (math.sqrt(3.)/6.) + dy, alat_cell * math.sqrt(2. / 3.) + dz), symbols=str(element))

    return structure


def hcp0001_inter(element, alat, hs_vect):
    """
    Workfunction to create an hcp(0001) interface with a given element for the calculation of the adhesion energy.

    :param element: The element to create the structure with.
    :return: The structure.
    """
    hs_np_vect = hs_vect.get_array('hs_array')
    alat_cell = alat
    cell_unit = array([[1.0, 0., 0.], [-0.5, math.sqrt(3) / 2., 0.], [0., 0., 2 * math.sqrt(2. / 3.)]]) * alat_cell
    rip_vet = array([1, 1, 3])
    the_cell = cell_unit*rip_vet                     
    the_cell[2][2] = 2 * the_cell[2][2] + vacuum_slab
    StructureData = DataFactory("structure")
    structure = StructureData(cell=the_cell)
    for nx in arange(rip_vet[0]):
        for ny in arange(rip_vet[1]):
            for nz in arange(rip_vet[2]):
                dx = nx*cell_unit[0][0] + ny*cell_unit[1][0] + nz*cell_unit[2][0]
                dy = nx*cell_unit[0][1] + ny*cell_unit[1][1] + nz*cell_unit[2][1]
                dz = nx*cell_unit[0][2] + ny*cell_unit[1][2] + nz*cell_unit[2][2]
                structure.append_atom(position=(alat_cell * 0. + dx + hs_np_vect[0], alat_cell * 0. + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(2., -1) + alat_cell * 0. + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx + hs_np_vect[0], alat_cell * (math.sqrt(3.)/6.) + dy + hs_np_vect[1], inter_factor*cell_unit[2][2] * math.pow(2., -1) + alat_cell * math.sqrt(2. / 3.) + dz), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0. + dx, -1 * ( alat_cell * 0. + dy), -1 * ( alat_cell * 0. + dz) ), symbols=str(element))
                structure.append_atom(position=(alat_cell * 0.5 + dx, -1 * ( alat_cell * (math.sqrt(3.)/6.) + dy), -1 * ( alat_cell * math.sqrt(2. / 3.) + dz) ), symbols=str(element))

    return structure

