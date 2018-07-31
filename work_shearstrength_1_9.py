#-*- coding: utf-8 -*-
from aiida.backends.utils import load_dbenv, is_dbenv_loaded

__license__ = "MIT license, see LICENSE.txt file."
__authors__ = "Tribology Group of the University of Modena."
__version__ = "0.9.1"

if not is_dbenv_loaded():
    load_dbenv()

from aiida.orm import load_node
from aiida.orm.utils import DataFactory
from aiida.work.db_types import Int, Float, Str, Bool, BaseType
from aiida.orm.data.base import NumericType
from aiida.orm.code import Code
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array import ArrayData
from aiida.work.run import run, submit
from aiida.work.workchain import (WorkChain, ToContext, while_)
from aiida.work.workfunction import workfunction
from aiida.orm.calculation.job.quantumespresso.pw import PwCalculation
from functions_shearstrength_1_9 import *
from structures_shearstrength_1_9 import *

from matplotlib import use
use('Agg')
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
import os
import json
#from numpy import array, arange, append, linspace, amin, concatenate, mgrid, shape, zeros, float
from scipy import interpolate
from scipy.interpolate import interp1d, interp2d, griddata, Rbf

# Set up the factories
ParameterData = DataFactory("parameter")
KpointsData = DataFactory("array.kpoints")
PwProcess = PwCalculation.process()

class ShearStrengthWF(WorkChain):
    """
    General workflow that finds the optimized lattice parameter of the selected element via Newton's algorithm on the first derivative of the energy (minus the pressure), calculates the surface energy, creates an interface to find the adhesion energy.
    """

    @classmethod
    def define(cls, spec):
        """
        Workfunction definition
        """
        super(ShearStrengthWF, cls).define(spec)
        spec.outline(
	    cls.read_inputs,
	    cls.init_functions,
            cls.init_optlat,
            cls.put_step0_in_ctx,
            cls.move_next_step,
            while_(cls.not_converged_alat)(
                cls.move_next_step,
            ),
            cls.report_optlat,
	    cls.kpoints_conv,
	    cls.run_bulkslab,
            cls.report_surfen,
            while_(cls.not_converged_inter)(
                cls.run_inter,
                cls.report_inter,
	    	cls.accumulate_all,
                cls.increase_iter
                ),
            cls.make_chargediff_pw,
            cls.make_chargediff_pp_min,
            cls.average_chargediff_min,
            cls.make_chargediff_pp_max,
            cls.average_chargediff_max,
            cls.report_chargediff,
            cls.compute_PPES,
            cls.report_worksep,
	    cls.plot_pes
        )
        spec.dynamic_output()

    def read_inputs(self):
	"""
	Reads the input parameters from the file 'workflow_input.in' and
	converts them to AiiDA base types. If essential input is missing the
	workflow will be terminated. For non-essential options defaults will
	be used.
	The input file may contain empty lines and comments, the order in which
	the parameters are given is not important. A check if the parameter names
	are known is performed.
	"""
	
	try:
	    inputfile = open('workflow_input.in', 'r')
	except IOError:
            with open('output.txt', 'a') as out:
                out.write(" ")
                out.write("Error opening input file 'workflow_input.in'. Please make sure that it is in the current directory.\n")
                out.write(" ")
	    raise SystemExit
	#setup a dictionary to read the data into. This makes it possible to change the order of lines.
	input_dict={}
	for line in inputfile:
	    #remove any leading blanks and check for comments or empty lines:
	    line.strip()
	    if not line.startswith('#') and not line.startswith('\n'):
		#remove comments and also trailing blanks
	        line = line.partition('#')[0]
	        line = line.rstrip()
		#partition the data in key and value pairs and put them in a dictionary
	        name, var = line.partition("=")[::2]
	        input_dict[name.strip()] = var

	inputfile.close()

	#the following keys are allowed variables. Essential keys must be set and will be checked.
	essential_keys = ['element', 'lattice', 'lattice_constant', 'selected_surface', 'use_vdW', 'is_metal', 'is_magnetic', 'pw_code', 'pp_code', 'pseudopotential_family']
	additional_keys = ['relative_tolerance_volume', 'energy_tolerance', 'scf_wallclock_max_time', 'relax_wallclock_max_time', 'charge_cutoff_multiplier', 'vdW_scaling', 'bands']
	known_keys = essential_keys + additional_keys
	#initialize checking dictionary for essential inputs
	check_essential = {}
	for key in essential_keys:
		check_essential[key] = False
	for key in input_dict.keys():
		#Check for unknown parameters:
		if key not in known_keys:
                        with open('output.txt', 'a') as out:
                            out.write(" ")
                            out.write("The input parameter "'+key+'" is not known. Please check your input file and use only the following parameters:\n")
                            out.writelines(["{0} ".format(item for item in known_keys)])
                            out.write("\n")
			raise SystemExit
		elif key == 'element':
			self.ctx.element = str(input_dict[key].strip())
			check_essential[key] = True
		elif key == 'lattice':
			self.ctx.lattice = str(input_dict[key].strip())
			check_essential[key] = True
		elif key == 'lattice_constant':
			self.ctx.alat = Float(input_dict[key])
			check_essential[key] = True
		elif key == 'selected_surface':
			self.ctx.miller = str(input_dict[key].strip())
			check_essential[key] = True
		elif key == 'use_vdW':
			self.ctx.vdw = str_to_bool(str(input_dict[key].strip()))
			check_essential[key] = True
		elif key == 'is_metal':
			self.ctx.metal = str_to_bool(str(input_dict[key].strip()))
			check_essential[key] = True
		elif key == 'is_magnetic':
			self.ctx.magnet = str_to_bool(str(input_dict[key].strip()))
			check_essential[key] = True
		elif key == 'pw_code':
			self.ctx.code = Str(input_dict[key].strip())
			check_essential[key] = True
		elif key == 'pp_code':
			self.ctx.ppcode = Str(input_dict[key].strip())
			check_essential[key] = True
		elif key == 'pseudopotential_family':
			self.ctx.pseudo_family = Str(input_dict[key].strip())
			check_essential[key] = True
	"""
	In the following the additional keys are handled and default values are set. Be careful when editing the defaults!
        """
	for key in additional_keys:
		if key == 'relative_tolerance_volume':
			if key in input_dict:
				self.ctx.volume_tolerance = Float(input_dict[key])
		        else:
				self.ctx.volume_tolerance = Float(0.0005)
				
		if key == 'energy_tolerance':
			if key in input_dict:
				self.ctx.energy_tolerance = Float(input_dict[key])
		        else:
				self.ctx.energy_tolerance = Float(0.001)
				
		if key ==  'bands':
			if key in input_dict:
				self.ctx.bands = int(input_dict[key])
		        else:
				self.ctx.bands = int(0) #The default is NOT resulting in 0 bands, but enables QE to choose the number of bands itself
				
		if key ==  'scf_wallclock_max_time':
			if key in input_dict:
				self.ctx.scf_wallclock_max_time = int(input_dict[key])
		        else:
				self.ctx.scf_wallclock_max_time = int(1500)
	
		if key ==  'relax_wallclock_max_time':
			if key in input_dict:
				self.ctx.relax_wallclock_max_time = int(input_dict[key])
		        else:
				self.ctx.relax_wallclock_max_time = int(7200)
				
		if key ==  'charge_cutoff_multiplier':
			if key in input_dict:
				self.ctx.charge_cutoff_multiplier = Float(input_dict[key])
		        else:
				self.ctx.charge_cutoff_multiplier = Float(12.0)
				
		if key ==  'vdW_scaling':
			if key in input_dict:
				self.ctx.vdW_scaling = Float(input_dict[key])
		        else:
				self.ctx.vdW_scaling = Float(0.75)


	#check if all essential keys are present and print out missing ones if needed.
	for key in check_essential.keys():
		if check_essential[key] == False:
                        with open('output.txt', 'a') as out:
                             out.write("The essential input parameter "'+key+'" is missing. Please check your input file and add the appropriate line.\n")
	if not all(check_essential.values()):
		with open('output.txt', 'a') as out:
                    out.write(" ")
		raise SystemExit

	return

    def init_functions(self):

        cube_func = self.ctx.lattice
        bulk_func = self.ctx.lattice + self.ctx.miller + '_bulk'          
        slab_func = self.ctx.lattice + self.ctx.miller + '_slab'          
        inter_func = self.ctx.lattice + self.ctx.miller + '_inter'        
                                                                      
        cube_generate = globals()[cube_func]
        bulk_generate = globals()[bulk_func]
        slab_generate = globals()[slab_func]
        inter_generate = globals()[inter_func]

        hsarray = 'hsarray_' + self.ctx.lattice + self.ctx.miller
        
        global cube_generate, bulk_generate, slab_generate, inter_generate, hsarray
        
        #return

    def init_optlat(self):
        """
	Create two initial structures with different volume
	(the second volume is increased by 4 angstrom^3).
	Converge the energy cutoff for the wavefunctions by computing
	the energy difference between this structures for increasing
	cutoff.
        Then launch the first calculations for the lattice constant optimization
        and store the outputs of the two calcs in r0 and r1
        """
        
        global cube_generate
        structure_cube = cube_generate(element=Str(self.ctx.element), alat=Float(self.ctx.alat))

        initial_volume = structure_cube.get_cell_volume()
        new_volume = initial_volume + 4.  # In ang^3

        scaled_structure = get_new_structure(structure_cube, new_volume)
        
	Starting_EncutWF = 15.0
	Encut_WF_increment = 2.5

	Encut_WF = converge_Encut(Float(Starting_EncutWF), Float(Encut_WF_increment), Float(self.ctx.charge_cutoff_multiplier), Float(self.ctx.energy_tolerance), structure_cube, scaled_structure, Str(self.ctx.pseudo_family), Str(self.ctx.code), Bool(self.ctx.metal), Bool(self.ctx.magnet), Bool(self.ctx.vdw), Int(self.ctx.scf_wallclock_max_time), Float(self.ctx.vdW_scaling))

	self.ctx.energy_cutoff_wavefunction = Encut_WF
	self.ctx.energy_cutoff_charge = Encut_WF * self.ctx.charge_cutoff_multiplier

	with open('output.txt', 'a') as out:
		out.write('The converged value for the energy cutoff for the wavefunctions is '+str(self.ctx.energy_cutoff_wavefunction)+'Ry.\n')
		out.write('The corresponding cutoff for the charge density (due to the multiplier of '+str(self.ctx.charge_cutoff_multiplier)+') is '+str(self.ctx.energy_cutoff_charge)+'Ry.\n')
		out.write('\n')

	inputs1 = generate_cube_input_params(scaled_structure, self.ctx.code, self.ctx.pseudo_family, self.ctx.metal, self.ctx.magnet, self.ctx.vdw, self.ctx.scf_wallclock_max_time, self.ctx.energy_cutoff_wavefunction, self.ctx.energy_cutoff_charge, self.ctx.vdW_scaling)
        inputs0 = generate_cube_input_params(structure_cube, self.ctx.code, self.ctx.pseudo_family, self.ctx.metal, self.ctx.magnet, self.ctx.vdw, self.ctx.scf_wallclock_max_time, self.ctx.energy_cutoff_wavefunction, self.ctx.energy_cutoff_charge, self.ctx.vdW_scaling)

        self.ctx.last_structure = scaled_structure

        # Run PW
        future0 = submit(PwProcess, **inputs0)
        future1 = submit(PwProcess, **inputs1)

        # Wait to complete before next step
        return ToContext(r0=future0, r1=future1)

    def put_step0_in_ctx(self):
        """
        Store the outputs of the very first step in a specific dictionary
        """

        #V, E, dE = get_volume_energy_and_derivative(self.ctx.r0['output_parameters'])
        V, E, dE = get_volume_energy_and_derivative(self.ctx.r0.out.output_parameters)

        self.ctx.step0 = {'V': V, 'E': E, 'dE': dE}

        # Prepare the list containing the steps
        # step 1 will be stored here by move_next_step
        self.ctx.steps = []

    def move_next_step(self):
        """
        Main routine that reads the two previous calculations r0 and r1,
        uses the Newton's algorithm on the pressure (i.e., fits the results
        with a parabola and sets the next point to calculate to the parabola
        minimum).
        r0 gets replaced with r1, r1 will get replaced by the results of the
        new calculation.
        """

        global cube_generate
        ddE = get_second_derivative(self.ctx.r0.out.output_parameters,
                                    self.ctx.r1.out.output_parameters)
        V, E, dE = get_volume_energy_and_derivative(self.ctx.r1.out.output_parameters)
        a, b, c = get_abc(V, E, dE, ddE)

        new_step_data = {'V': V, 'E': E, 'dE': dE, 'ddE': ddE,
                         'a': a, 'b': b, 'c': c}
        self.ctx.steps.append(new_step_data)

        # Minimum of a parabola
        new_volume = -b / 2. / a

        # remove older step
        self.ctx.r0 = self.ctx.r1
        structure_cube = cube_generate(element=Str(self.ctx.element), alat=Float(self.ctx.alat))
        scaled_structure = get_new_structure(structure_cube, new_volume)
        self.ctx.last_structure = scaled_structure

        inputs = generate_cube_input_params(scaled_structure, self.ctx.code, self.ctx.pseudo_family, self.ctx.metal, self.ctx.magnet, self.ctx.vdw, self.ctx.scf_wallclock_max_time, self.ctx.energy_cutoff_wavefunction, self.ctx.energy_cutoff_charge, self.ctx.vdW_scaling)

        # Run PW
        future = submit(PwProcess, **inputs)
        # Replace r1
        return ToContext(r1=future)

    def not_converged_alat(self):
        """
        Return True if the worflow is not converged yet (i.e., the volume changed significantly)
        """
        return abs(self.ctx.r1.out.output_parameters.dict.volume - self.ctx.r0.out.output_parameters.dict.volume)/(self.ctx.r0.out.output_parameters.dict.volume) > self.ctx.volume_tolerance

    def report_optlat(self):
        """
        Output final quantities
        """
	if self.ctx.lattice == 'graphene' or self.ctx.lattice == 'hcp':
   	    opt_alat = optalat_hex(self.ctx.last_structure)
        else:
            opt_alat = optalat(self.ctx.last_structure)
        with open('output.txt', 'a') as out:
            out.write("lattice parameter  =  {0}  angstrom\n".format(str(opt_alat)))

    def kpoints_conv(self):
	"""
	Test
	"""
	from aiida.orm import DataFactory
	if self.ctx.lattice == 'graphene' or self.ctx.lattice == 'hcp':
   	    opt_alat = optalat_hex(self.ctx.last_structure)
        else:
            opt_alat = optalat(self.ctx.last_structure)

        global bulk_generate
        structure_bulk = bulk_generate(element=Str(self.ctx.element), alat=Float(opt_alat))	
        
        with open('output.txt', 'a') as out:
            out.write('Package number of the bulk structure = '+str(structure_bulk.pk))
            out.write("\n")
#        print structure_bulk

	k_distance = converge_kpts(structure_bulk, self.ctx.energy_tolerance, self.ctx.code, self.ctx.pseudo_family, Bool(self.ctx.metal), Bool(self.ctx.magnet), Bool(self.ctx.vdw), Int(self.ctx.relax_wallclock_max_time), self.ctx.energy_cutoff_wavefunction, self.ctx.energy_cutoff_charge, self.ctx.vdW_scaling)

	global k_distance
        
        with open('output.txt', 'a') as out:
            out.write("kpoints distance  =  {0}  angstrom^-1\n".format(str(k_distance))) 
        
#	print "kpoints distance = ", k_distance, "angstrom^-1"

    def run_bulkslab(self):
        """
        Initialize bulk and slab structures with the optimized lattice parameter and lunch the bulk and slab calculations
        """

	#generate bulk and slab structure and the related input
        from aiida.orm import DataFactory
	if self.ctx.lattice == 'graphene' or self.ctx.lattice == 'hcp':
   	    opt_alat = optalat_hex(self.ctx.last_structure)
        else:
            opt_alat = optalat(self.ctx.last_structure)

        global bulk_generate, slab_generate
        structure_bulk = bulk_generate(element=Str(self.ctx.element), alat=Float(opt_alat))

    	structure_slab = slab_generate(element=Str(self.ctx.element), alat=Float(opt_alat))
        self.ctx.structureslab = structure_slab

        input_bulk = generate_bulk_input_params(structure_bulk, self.ctx.code, self.ctx.pseudo_family, self.ctx.metal, self.ctx.magnet, self.ctx.vdw, k_distance, self.ctx.relax_wallclock_max_time, self.ctx.energy_cutoff_wavefunction, self.ctx.energy_cutoff_charge, self.ctx.vdW_scaling)
	input_slab = generate_slab_input_params(structure_slab, self.ctx.code, self.ctx.pseudo_family, self.ctx.metal, self.ctx.magnet, self.ctx.bands, self.ctx.vdw, k_distance, self.ctx.relax_wallclock_max_time, self.ctx.energy_cutoff_wavefunction, self.ctx.energy_cutoff_charge, self.ctx.vdW_scaling)
        
        # Launch the calculations
        future_bulk = submit(PwProcess, **input_bulk)
        future_slab = submit(PwProcess, **input_slab)
        
	#Store the results
        return ToContext(result_bulk=future_bulk, result_slab=future_slab)

    def report_surfen(self):
        """
        Output the surface energy
        """
        E_bulk = Float(get_energy(self.ctx.result_bulk.out.output_parameters))
        E_slab = Float(get_energy(self.ctx.result_slab.out.output_parameters))
        N_bulk = self.ctx.result_bulk.out.output_parameters.dict.number_of_atoms
        N_slab = self.ctx.result_slab.out.output_parameters.dict.number_of_atoms

	if self.ctx.lattice == 'bcc' and self.ctx.miller == '110':
   	    area = Float(2*self.ctx.structureslab.cell[0][0]*self.ctx.structureslab.cell[1][1])
        else:
            area = Float(self.ctx.structureslab.cell[0][0]*self.ctx.structureslab.cell[1][1])

        N_ratio = Float(N_slab) * math.pow(Float(N_bulk), -1)
        E_surf = surfen(E_slab, E_bulk, N_ratio, area)
        with open('output.txt', 'a') as out:
            out.write("Surface Energy  =  {0}  J/m2; corresponding calculation has package number  = {1} \n".format(str(E_surf), str(self.ctx.result_slab.pk)))
#        print "Surface Energy = ", E_surf, "J/m2"

    def not_converged_inter(self):
        """
        Return True if the worflow is not converged yet (i.e., the calculation along the hsarray is not completed)
        """
        
        global hsarray
        from aiida.orm import DataFactory
	if self.ctx.lattice == 'graphene' or self.ctx.lattice == 'hcp':
   	    opt_alat = optalat_hex(self.ctx.last_structure)
        else:
            opt_alat = optalat(self.ctx.last_structure)

        hs_dictionary = hs_dict(alat=Float(opt_alat))

        return iter_adh < len(hs_dictionary[hsarray])

    def run_inter(self):
        """
        Run the calculation of the system containing the interface
        """

        global hsarray
        from aiida.orm import DataFactory
	if self.ctx.lattice == 'graphene' or self.ctx.lattice == 'hcp':
   	    opt_alat = optalat_hex(self.ctx.last_structure)
        else:
            opt_alat = optalat(self.ctx.last_structure)

        global inter_generate, hs_array
	hs_dictionary = hs_dict(alat=Float(opt_alat))

        hs_array = ArrayData()
        hs_array.set_array('hs_array', hs_dictionary[hsarray][iter_adh])

	structure_inter = inter_generate(element=Str(self.ctx.element), alat=Float(opt_alat), hs_vect=hs_array)
        input_inter = generate_inter_input_params(structure_inter, self.ctx.code, self.ctx.pseudo_family, self.ctx.metal, self.ctx.magnet, self.ctx.bands, self.ctx.vdw, k_distance, self.ctx.relax_wallclock_max_time, self.ctx.energy_cutoff_wavefunction, self.ctx.energy_cutoff_charge, self.ctx.vdW_scaling)
      
        future_inter = submit(PwProcess, **input_inter)
        
        return ToContext(result_inter=future_inter)

    def report_inter(self):

        global hsarray
        from aiida.orm import DataFactory
	if self.ctx.lattice == 'graphene' or self.ctx.lattice == 'hcp':
   	    opt_alat = optalat_hex(self.ctx.last_structure)
        else:
            opt_alat = optalat(self.ctx.last_structure)

	hs_dictionary = hs_dict(alat=Float(opt_alat))
        hs_array = ArrayData()
        hs_array.set_array('hs_array', hs_dictionary[hsarray][iter_adh])
        E_slab = Float(get_energy(self.ctx.result_slab.out.output_parameters))
        E_inter = Float(get_energy(self.ctx.result_inter.out.output_parameters))

	if self.ctx.lattice == 'bcc' and self.ctx.miller == '110':
   	    area = Float(2*self.ctx.structureslab.cell[0][0]*self.ctx.structureslab.cell[1][1])
        else:
            area = Float(self.ctx.structureslab.cell[0][0]*self.ctx.structureslab.cell[1][1])

        E_adh = interen(E_inter, E_slab, area)
        with open('output.txt', 'a') as out:
            out.write("Position  {0}  Adhesion Energy  =  {1}  J/m2; corresponding calculation package number is: {2} \n".format(str(hs_dictionary[hsarray][iter_adh]), str(E_adh), str(self.ctx.result_inter.pk)))
#        print "Position ", hs_dictionary[hsarray][iter_adh], "  Adhesion Energy = ", E_adh, "J/m2"

    def accumulate_all(self):

	global array_accumulo, hsarray
        from aiida.orm import DataFactory
	if self.ctx.lattice == 'graphene' or self.ctx.lattice == 'hcp':
   	    opt_alat = optalat_hex(self.ctx.last_structure)
        else:
            opt_alat = optalat(self.ctx.last_structure)

	hs_dictionary = hs_dict(alat=Float(opt_alat))

	x_cord = hs_dictionary[hsarray][iter_adh][0]
	y_cord = hs_dictionary[hsarray][iter_adh][1]

	if self.ctx.lattice == 'bcc' and self.ctx.miller == '110':
	    E_adh = (Float(get_energy(self.ctx.result_inter.out.output_parameters)) - 2 * Float(get_energy(self.ctx.result_slab.out.output_parameters))) * math.pow(Float(2*self.ctx.structureslab.cell[0][0]*self.ctx.structureslab.cell[1][1]), -1) * eVA2_to_Jm2
        else:
	    E_adh = (Float(get_energy(self.ctx.result_inter.out.output_parameters)) - 2 * Float(get_energy(self.ctx.result_slab.out.output_parameters))) * math.pow(Float(self.ctx.structureslab.cell[0][0]*self.ctx.structureslab.cell[1][1]), -1) * eVA2_to_Jm2

	list_all = [iter_adh, x_cord, y_cord, E_adh, self.ctx.result_inter.pk]

	array_accumulo.append(list_all)

	#return array_accumulo

    def increase_iter(self):

        global iter_adh   #self.iter_adh
        iter_adh += 1
        #return iter_adh

    def make_chargediff_pw(self):
        
        global array_accumulo
        sort_array = sorted(array_accumulo,key=lambda x: x[3])
        self.ctx.min_calc = load_node(sort_array[0][4])
        self.ctx.max_calc = load_node(sort_array[-1][4])
        min_structure = self.ctx.min_calc.out.output_structure
        max_structure = self.ctx.max_calc.out.output_structure

        min_low_struct = make_lower_slab(min_structure)
        min_upp_struct = make_upper_slab(min_structure)
        input_min_low = generate_scf_input_params(min_low_struct, self.ctx.code, self.ctx.pseudo_family, self.ctx.metal, self.ctx.magnet, self.ctx.bands, self.ctx.vdw, k_distance, self.ctx.scf_wallclock_max_time, self.ctx.energy_cutoff_wavefunction, self.ctx.energy_cutoff_charge, self.ctx.vdW_scaling)
        input_min_upp = generate_scf_input_params(min_upp_struct, self.ctx.code, self.ctx.pseudo_family, self.ctx.metal, self.ctx.magnet, self.ctx.bands, self.ctx.vdw, k_distance, self.ctx.scf_wallclock_max_time, self.ctx.energy_cutoff_wavefunction, self.ctx.energy_cutoff_charge, self.ctx.vdW_scaling)
      
        max_low_struct = make_lower_slab(max_structure)
        max_upp_struct = make_upper_slab(max_structure)
        input_max_low = generate_scf_input_params(max_low_struct, self.ctx.code, self.ctx.pseudo_family, self.ctx.metal, self.ctx.magnet, self.ctx.bands, self.ctx.vdw, k_distance, self.ctx.scf_wallclock_max_time, self.ctx.energy_cutoff_wavefunction, self.ctx.energy_cutoff_charge, self.ctx.vdW_scaling)
        input_max_upp = generate_scf_input_params(max_upp_struct, self.ctx.code, self.ctx.pseudo_family, self.ctx.metal, self.ctx.magnet, self.ctx.bands, self.ctx.vdw, k_distance, self.ctx.scf_wallclock_max_time, self.ctx.energy_cutoff_wavefunction, self.ctx.energy_cutoff_charge, self.ctx.vdW_scaling)

        future_min_low = submit(PwProcess, **input_min_low)
        future_min_upp = submit(PwProcess, **input_min_upp)
        future_max_low = submit(PwProcess, **input_max_low)
        future_max_upp = submit(PwProcess, **input_max_upp)
        
        return ToContext(min_calc_low=future_min_low, min_calc_upp=future_min_upp, max_calc_low=future_max_low, max_calc_upp=future_max_upp)

    def make_chargediff_pp_min(self):
        
        parent_tot = self.ctx.min_calc.out.remote_folder
        parent_upp = self.ctx.min_calc_low.out.remote_folder
        parent_low = self.ctx.min_calc_upp.out.remote_folder

        self.ctx.chargefiles_min = run_slab_pp_async(parent_tot,parent_upp,parent_low,self.ctx.ppcode)

    def make_chargediff_pp_max(self):
        
        parent_tot = self.ctx.max_calc.out.remote_folder
        parent_upp = self.ctx.max_calc_low.out.remote_folder
        parent_low = self.ctx.max_calc_upp.out.remote_folder

        self.ctx.chargefiles_max = run_slab_pp_async(parent_tot,parent_upp,parent_low,self.ctx.ppcode)

    def average_chargediff_min(self):

        self.ctx.chargediff_results_min = compute_charge_density_difference(self.ctx.chargefiles_min,Bool(1))
	os.rename('RhoDiff.png', 'RhoDiff_min.png')

    def average_chargediff_max(self):

        self.ctx.chargediff_results_max = compute_charge_density_difference(self.ctx.chargefiles_max,Bool(1))
	os.rename('RhoDiff.png', 'RhoDiff_max.png')

    def report_chargediff(self):
	
	with open('Charge_diff_out.txt', 'w') as out:
		out.write(' Output dictionary for minimum configuration:\n')
		out.write('')
		out.write(json.dumps(self.ctx.chargediff_results_min.get_dict(),sort_keys=True,indent=4))
		out.write('')
		out.write('')
		out.write(' Output dictionary for maximum configuration:\n')
		out.write('')
		out.write(json.dumps(self.ctx.chargediff_results_max.get_dict(),sort_keys=True,indent=4))

    def compute_PPES(self):
        """
        Computes the perpendicular potential energy surface in the minimum
        configuration. Parallel and sequential execution are both supported.
        """

        dz_min=-1.0
        dz_max=3.1
        dz_step=0.25

        execute_in_parallel = False

        min_structure = self.ctx.min_calc.out.output_structure
        min_low_struct = make_lower_slab(min_structure)
        min_upp_struct = make_upper_slab(min_structure)

        max_structure = self.ctx.max_calc.out.output_structure
        max_low_struct = make_lower_slab(max_structure)
        max_upp_struct = make_upper_slab(max_structure)

        System_Formula = min_structure.get_formula()

        if execute_in_parallel:
                PPES_min = make_PPES_parallel(min_upp_struct, min_low_struct, Float(dz_min), Float(dz_max), Float(dz_step), Str(self.ctx.code), Str(self.ctx.pseudo_family), Bool(self.ctx.metal), Bool(self.ctx.magnet), Int(self.ctx.bands), Bool(self.ctx.vdw), Float(k_distance), Int(self.ctx.scf_wallclock_max_time), Float(self.ctx.energy_cutoff_wavefunction), Float(self.ctx.energy_cutoff_charge), Float(self.ctx.vdW_scaling))

                os.rename(System_Formula+"_PPES.png",System_Formula+"_Min_PPES.png")

                PPES_max = make_PPES_parallel(max_upp_struct, max_low_struct, Float(dz_min), Float(dz_max), Float(dz_step), Str(self.ctx.code), Str(self.ctx.pseudo_family), Bool(self.ctx.metal), Bool(self.ctx.magnet), Int(self.ctx.bands), Bool(self.ctx.vdw), Float(k_distance), Int(self.ctx.scf_wallclock_max_time), Float(self.ctx.energy_cutoff_wavefunction), Float(self.ctx.energy_cutoff_charge), Float(self.ctx.vdW_scaling))

                os.rename(System_Formula+"_PPES.png",System_Formula+"_Max_PPES.png")
        else:    
                PPES_min = make_PPES_sequential(min_upp_struct, min_low_struct, Float(dz_min), Float(dz_max), Float(dz_step), Str(self.ctx.code), Str(self.ctx.pseudo_family), Bool(self.ctx.metal), Bool(self.ctx.magnet), Int(self.ctx.bands), Bool(self.ctx.vdw), Float(k_distance), Int(self.ctx.scf_wallclock_max_time), Float(self.ctx.energy_cutoff_wavefunction), Float(self.ctx.energy_cutoff_charge), Float(self.ctx.vdW_scaling))

                os.rename(System_Formula+"_PPES.png",System_Formula+"_Min_PPES.png")

                PPES_max = make_PPES_sequential(max_upp_struct, max_low_struct, Float(dz_min), Float(dz_max), Float(dz_step), Str(self.ctx.code), Str(self.ctx.pseudo_family), Bool(self.ctx.metal), Bool(self.ctx.magnet), Int(self.ctx.bands), Bool(self.ctx.vdw), Float(k_distance), Int(self.ctx.scf_wallclock_max_time), Float(self.ctx.energy_cutoff_wavefunction), Float(self.ctx.energy_cutoff_charge), Float(self.ctx.vdW_scaling))

                os.rename(System_Formula+"_PPES.png",System_Formula+"_Max_PPES.png")
        
        with open('output.txt', 'a') as out:        
            out.write("Minimum P-PES array package number  =  {0}\n".format(str(PPES_min.pk)))
            out.write("Maximum P-PES array package number  =  {0}\n".format(str(PPES_max.pk)))
#        print('Minimum P-PES array package number = '+str(PPES_min.pk))
#        print('Maximum P-PES array package number = '+str(PPES_max.pk))

        PPES_min.label = 'P-PES array for min of '+str(System_Formula)
        PPES_max.label = 'P-PES array for max of '+str(System_Formula)
        PPES_min.description = 'Output P-PES array at minimum position for '+str(System_Formula)+' consisting of '+str(len(PPES_min.get_array('PPES_array')))+' (z-shift, tot_Energy) data pairs.'
        PPES_max.description = 'Output P-PES array at maximum position for '+str(System_Formula)+' consisting of '+str(len(PPES_max.get_array('PPES_array')))+' (z-shift, tot_Energy) data pairs.'

    def report_worksep(self):
        
        global array_accumulo
        array_accumulo = np.array(array_accumulo, dtype=np.float64)
        wsep = Float(-np.amin(array_accumulo[:,3]))
        W_sep = workofsep(wsep)
        with open('output.txt', 'a') as out:
            out.write("Work of separation  =  {0}  J/m2\n".format(str(W_sep)))
#        print "Work of separation = ", W_sep, "J/m2"

    def plot_pes(self):

	global array_accumulo
	array_accumulo = np.array(array_accumulo, dtype=np.float64)
	
        wsep_row = array_accumulo[np.argmin(array_accumulo[:,3]),:] 
        array_accumulo[:,1:4] -= wsep_row[1:4]

	support = array_accumulo.copy()
	replica = array_accumulo.copy()
	
	alat_x = self.ctx.structureslab.cell[0][0]
	alat_y = self.ctx.structureslab.cell[1][1]

	if self.ctx.lattice == 'bcc' and self.ctx.miller == '110':
            alat_x = alat_x*2.0
            alat_y = alat_y*2.0

	if self.ctx.miller == '111' or self.ctx.miller == '0001':
            alat_y = alat_y*2.0
        
        if self.ctx.lattice+self.ctx.miller in ['fcc100', 'bcc100']:
            array_accumulo = replica_for_pes_100(replica, support, array_accumulo, alat_x, alat_y)
        elif self.ctx.lattice+self.ctx.miller == 'diamond100':
	    array_accumulo = replica_for_pes_dia100(replica, support, array_accumulo, alat_x, alat_y)
        elif self.ctx.lattice+self.ctx.miller == 'fcc110':
            array_accumulo = replica_for_pes_fcc110(replica, support, array_accumulo, alat_x, alat_y)
        elif self.ctx.lattice+self.ctx.miller == 'bcc110':
            array_accumulo = replica_for_pes_bcc110(replica, support, array_accumulo, alat_x, alat_y)
        elif self.ctx.lattice+self.ctx.miller == 'diamond110':
            array_accumulo = replica_for_pes_dia110(replica, support, array_accumulo, alat_x, alat_y)
        elif self.ctx.miller == '111' or self.ctx.miller == '0001':
            array_accumulo = replica_for_pes_111(replica, support, array_accumulo, alat_x, alat_y)
        
        support = array_accumulo.copy()
	replica = array_accumulo.copy()
        
        for i in np.arange(1,5):
            replica[:,1] = support[:,1]+alat_x*i
            replica[:,2] = support[:,2]+0
            array_accumulo = np.concatenate((array_accumulo, replica), axis=0)

        for i in np.arange(1,5):
            replica[:,1] = support[:,1]+0
            replica[:,2] = support[:,2]+alat_y*i
            array_accumulo = np.concatenate((array_accumulo, replica), axis=0)

        for i in np.arange(1,5):
            for j in np.arange(1,5):
                replica[:,1] = support[:,1]+alat_x*i
                replica[:,2] = support[:,2]+alat_y*j
                array_accumulo = np.concatenate((array_accumulo, replica), axis=0)
        
        array_accumulo[:,1]-=alat_x*2
        array_accumulo[:,2]-=alat_y*2
        
        np.savetxt('array_for_2dpes.dat', array_accumulo)
        
	label = np.array(array_accumulo[:,0])
        x = np.array(array_accumulo[:,1])
        y = np.array(array_accumulo[:,2])
        E = np.array(array_accumulo[:,3])

        fact = 1.

        if self.ctx.miller == '100':
            xnew, ynew = np.mgrid[-fact*alat_x :fact*alat_x:260j,-fact*alat_y:fact*alat_y:260j]
        if self.ctx.miller == '110':
            xnew, ynew = np.mgrid[-fact*alat_x :fact*alat_x:220j,-fact*alat_y:fact*alat_y:312j]
        if self.ctx.miller == '111' or self.ctx.miller == '0001':
            xnew, ynew = np.mgrid[-fact*alat_x :fact*alat_x:200j,-fact*alat_y:fact*alat_y:346j]
            
        rbf = Rbf(x, y, E, function='cubic')
        
        Enew=rbf(xnew, ynew)

        level= 43
        fig = plt.figure(figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        anglerot='vertical'
        shrin=1.
        zt1=plt.contourf(xnew, ynew, Enew, level, extent=(-fact*alat_x,fact*alat_x,-fact*alat_y,fact*alat_y), cmap=plt.cm.RdYlBu_r)
        cbar1=plt.colorbar(zt1,ax=ax,orientation=anglerot,shrink=shrin)
        cbar1.set_label(r'$E_{adh} (J/m^2)$', rotation=270, labelpad=20,fontsize=15,family='serif')
        plt.title("PES for " + self.ctx.element + self.ctx.miller, fontsize=18,family='serif')
        ax.quiver(0. , 0., 1., 0.,scale=1.,scale_units='inches',width=0.01,color='white')
        ax.quiver(0. , 0., 0., 1.,scale=1.,scale_units='inches',width=0.01,color='white')
        ax.plot(0.,0.,'w.',ms=7)
        ax.text(0.5,-0.5,'[1 0 1]',rotation='horizontal',color='white', fontsize=14)
        ax.text(-0.5,1.,'[1 2 1]',rotation='vertical',color='white', fontsize=14)
        ax.axis([-fact*alat_x,fact*alat_x,-fact*alat_y,fact*alat_y])
        plt.xlabel(r"distance ($\AA$)",fontsize=12,family='serif')
        plt.ylabel(r"distance ($\AA$)",fontsize=12,family='serif')
        plt.savefig("PES" + self.ctx.element + self.ctx.miller + ".pdf")

        x_alongx=np.arange(-1.5*alat_x,1.5*alat_x,alat_x/300)
        y_alongx=np.zeros(len(x_alongx))
        y_alongy=np.arange(-1.5*alat_y,1.5*alat_y,alat_y/300)
        x_alongy=np.zeros(len(y_alongy))
                
        zdev_alongx=np.zeros( (len(x_alongx),2) )
        zdev_alongy=np.zeros( (len(y_alongy),2) )
        delta=0.0001
        
        for i in range(len(x_alongx)):
            coordx=x_alongx[i]
            coordy=y_alongx[i]
            coordx=coordx+delta
            tempValp=rbf(coordx,coordy)
            coordx=coordx - 2.*delta
            tempValm=rbf(coordx,coordy)
            zdev_alongx[i,0]=0.5*(tempValp-tempValm)/delta
            coordx=coordx+delta
            coordy=coordy+delta
            tempValp=rbf(coordx,coordy)
            coordy=coordy - 2.*delta
            tempValm=rbf(coordx,coordy)
            zdev_alongx[i,1]=0.5*(tempValp-tempValm)/delta
        
        for i in range(len(y_alongy)):
            coordx=x_alongy[i]
            coordy=y_alongy[i]
            coordx=coordx+delta
            tempValp=rbf(coordx,coordy)
            coordx=coordx - 2.*delta
            tempValm=rbf(coordx,coordy)
            zdev_alongy[i,0]=0.5*(tempValp-tempValm)/delta
            coordx=coordx+delta
            coordy=coordy+delta
            tempValp=rbf(coordx,coordy)
            coordy=coordy  -2.*delta
            tempValm=rbf(coordx,coordy)
            zdev_alongy[i,1]=0.5*(tempValp-tempValm)/delta
        
        fig1, ( (s1,s2) ) = plt.subplots(nrows=2, ncols=1, figsize=(3, 6), dpi=100, sharex=False, sharey=True)
        fig1.subplots_adjust(left=0.25, bottom=0.12, right=0.99, top=0.95, wspace=0.25, hspace=0.25)
        plt.minorticks_on()
        s1.plot(x_alongx,rbf(x_alongx,y_alongx),'-',color='blue')  
        s1.set_xlim( (-fact*alat_x, fact*alat_x) )
        s1.set_xlabel(r"distance along x ($\AA$)",fontsize=12,family='serif')
        s1.set_ylabel(r'$E_{adh} (J/m^2)$',fontsize=12,family='serif')
        s2.plot(y_alongy,rbf(x_alongy,y_alongy),'-',color='blue')
        s2.set_xlim( (-fact*alat_y, fact*alat_y) )
        s2.set_xlabel(r"distance along y ($\AA$)",fontsize=12,family='serif')
        s2.set_ylabel(r'$E_{adh} (J/m^2)$',fontsize=12,family='serif')
        plt.savefig("PEP" + self.ctx.element + self.ctx.miller + ".pdf")
        with open('output.txt', 'a') as out:
            out.write("### {0} ({1})\n".format(str(self.ctx.element),str(self.ctx.miller)))
            out.write("Shear strength estimated along x  =  {0}  GPa \n".format(str(-(np.amin(zdev_alongx[:,0]))*10.0)))
            out.write("Shear strength estimated along y  =  {0}  GPa \n".format(str(-(np.amin(zdev_alongy[:,1]))*10.0)))
#        print("### "+ self.ctx.element+"("+self.ctx.miller+")")
#        print ("Shear strength estimated along x = ", -(np.amin(zdev_alongx[:,0]))*10.0, " GPa")
#        print ("Shear strength estimated along y = ", -(np.amin(zdev_alongy[:,1]))*10.0, " GPa")

        "Calculation of Minimum Energy Path (MEP)"
        with open('output.txt', 'a') as out:
            out.write("Starting (local) calculation of MEP. This might take a while...\n")

        facx=0.75
        facy=facx 
        xa =-facx*alat_x
        ya =-facy*alat_y
        xb = facx*alat_x
        yb = facy*alat_y    
        # time-step (limited by the ODE step on line 83 & 84 but independent of n1)
        h = 0.001
        # max number of iterations
        nstepmax = 99999
        # frequency of plotting
        nstepplot = 10
        # parameter used as stopping criterion
        tol1 = 1e-7;
        # number of images along the string (try from  n1 = 3 up to n1 = 1e4)
        n1 = 101
        g1 = np.linspace(0,1,n1)
        x1 = (xb-xa)*g1+xa
        y1 = np.zeros(n1)
        dx = x1 - np.roll(x1,1)
        dy = y1 - np.roll(y1,1)
        dx[0]=0
        dy[0]=0.
        lxy  = np.cumsum(np.sqrt(dx**2+dy**2))
        lxy /= lxy[n1-1]
        xf1 = interp1d(lxy,x1,kind='cubic')
        x1  =  xf1(g1)
        yf1 = interp1d(lxy,y1,kind='cubic')
        y1  =  yf1(g1)
        xi = x1.copy()
        yi = y1.copy()

        ## Main loop
        for nstep in range(nstepmax):
            # calculation of the x and y-components of the force, dVx and dVy respectively
            ee = rbf(x1,y1)
            #derivative of the potential
            x1 += delta
            tempValp=rbf(x1,y1)
            x1 -= 2.*delta
            tempValm=rbf(x1,y1)
            dVx = 0.5*(tempValp-tempValm)/delta
            x1 += delta
            y1 += delta
            tempValp=rbf(x1,y1)
            y1 -= 2.*delta
            tempValm=rbf(x1,y1)
            y1 += delta
            dVy = 0.5*(tempValp-tempValm)/delta

            x0 = x1.copy()
            y0 = y1.copy()
            # string steps:
            # 1. evolve
            xt = x1 - h*dVx
            yt = y1 - h*dVy
            # 2. derivative
            xt += delta
            tempValp=rbf(xt,yt)
            xt -= 2.*delta
            tempValm=rbf(xt,yt)
            dVxt = 0.5*(tempValp-tempValm)/delta
            xt += delta
            yt += delta
            tempValp=rbf(xt,yt)
            yt -= 2.*delta
            tempValm=rbf(xt,yt)
            yt += delta
            dVyt = 0.5*(tempValp-tempValm)/delta

            x1 -= 0.5*h*(dVx+dVxt)
            y1 -= 0.5*h*(dVy+dVyt)
            # 3. reparametrize  
            dx = x1-np.roll(x1,1)
            dy = y1-np.roll(y1,1)
            dx[0] = 0.
            dy[0] = 0.
            lxy  = np.cumsum(np.sqrt(dx**2+dy**2))
            lxy /= lxy[n1-1]
            xf1 = interp1d(lxy,x1,kind='cubic')
            x1  =  xf1(g1)
            yf1 = interp1d(lxy,y1,kind='cubic')
            y1  =  yf1(g1)
            tol = (np.linalg.norm(x1-x0)+np.linalg.norm(y1-y0))/n1
            if tol <= tol1:
               break
        with open('output.txt', 'a') as out:
            out.write("ZTS calculation starting along x with {0} images\n".format(str(n1)))
#        print('ZTS calculation starting along x with %d images' % n1)
        if tol > tol1 :
            with open('output.txt', 'a') as out:
                out.write("The calculation failed to converge after {0} iterations tol = {1}\n".format(str(nstep), str(tol)))
#            print('The calculation failed to converge after %d iterations tol=%.3g ' % (nstep,tol) )
        else :
            with open('output.txt', 'a') as out:
                out.write("The calculation terminated after {0} iterations tol = {1}\n".format(str(nstep), str(tol)))
#            print('The calculation terminated after %d iterations tol=%.3g' % (nstep,tol) )

        # second attempt
        x2 = np.zeros(n1)
        y2 = (yb-ya)*g1 +ya
        dx = x2 - np.roll(x2,1)
        dy = y2 - np.roll(y2,1)
        dx[0]=0
        dy[0]=0.
        lxy  = np.cumsum(np.sqrt(dx**2+dy**2))
        lxy /= lxy[n1-1]
        xf2 = interp1d(lxy,x2,kind='cubic')
        x2  =  xf2(g1)
        yf2 = interp1d(lxy,y2,kind='cubic')
        y2  =  yf2(g1)
        xj = x2.copy()
        yj = y2.copy()
        ## Main loop
        for nstep in range(nstepmax):
            # calculation of the x and y-components of the force, dVx and dVy respectively
            ee = rbf(x2,y2)
            #derivative of the potential
            x1 += delta
            tempValp=rbf(x1,y1)
            x1 -= 2.*delta
            tempValm=rbf(x1,y1)
            dVx = 0.5*(tempValp-tempValm)/delta
            x1 += delta
            y1 += delta
            tempValp=rbf(x1,y1)
            y1 -= 2.*delta
            tempValm=rbf(x1,y1)
            y1 += delta
            dVy = 0.5*(tempValp-tempValm)/delta

            x0 = x2.copy()
            y0 = y2.copy()
            # string steps:
            # 1. evolve
            xt = x2 - h*dVx
            yt = y2 - h*dVy
            # 2. derivative
            xt += delta
            tempValp=rbf(xt,yt)
            xt -= 2.*delta
            tempValm=rbf(xt,yt)
            dVxt = 0.5*(tempValp-tempValm)/delta
            xt += delta
            yt += delta
            tempValp=rbf(xt,yt)
            yt -= 2.*delta
            tempValm=rbf(xt,yt)
            yt += delta
            dVyt = 0.5*(tempValp-tempValm)/delta

            x2 -= 0.5*h*(dVx+dVxt)
            y2 -= 0.5*h*(dVy+dVyt)
            # 3. reparametrize  
            dx = x2-np.roll(x2,1)
            dy = y2-np.roll(y2,1)
            dx[0] = 0.
            dy[0] = 0.
            lxy  = np.cumsum(np.sqrt(dx**2+dy**2))
            lxy /= lxy[n1-1]
            xf2 = interp1d(lxy,x2,kind='cubic')
            x2  =  xf2(g1)
            yf2 = interp1d(lxy,y2,kind='cubic')
            y2  =  yf2(g1)
            tol = (np.linalg.norm(x2-x0)+np.linalg.norm(y2-y0))/n1
            if tol <= tol1:
               break
        with open('output.txt', 'a') as out:
            out.write("ZTS calculation starting along y with {0} images\n".format(n1))
#        print('ZTS calculation starting along y with %d images' % n1)
        if tol > tol1 :
            with open('output.txt', 'a') as out:
                out.write("The calculation failed to converge after {0} iterations tol = {1}\n".format(str(nstep),str(tol)))
#            print('The calculation failed to converge after %d iterations tol=%.3g ' % (nstep,tol) )
        else :
            with open('output.txt', 'a') as out:
                out.write("The calculation terminated after {0} iterations tol = {1}\n".format(str(nstep),str(tol)))
#            print('The calculation terminated after %d iterations tol=%.3g ' % (nstep,tol) )
     
        ## Output
        #print(dx,dy)
        fig3 = plt.figure(figsize=(7, 7), dpi=100)
        ax = fig3.add_subplot(111)
        ax.set_aspect('equal')
        anglerot='vertical'
        shrin=1.
        level=43

        if self.ctx.miller == '100':
            xnew, ynew = np.mgrid[-fact*alat_x :fact*alat_x:260j,-fact*alat_y:fact*alat_y:260j]
        if self.ctx.miller == '110':
            xnew, ynew = np.mgrid[-fact*alat_x :fact*alat_x:220j,-fact*alat_y:fact*alat_y:312j]
        if self.ctx.miller == '111' or self.ctx.miller == '0001':
            xnew, ynew = np.mgrid[-fact*alat_x :fact*alat_x:200j,-fact*alat_y:fact*alat_y:346j]

        e1=np.max( rbf(xnew, ynew))
        e2=np.min( rbf(xnew, ynew))
        zt1=plt.contourf(xnew, ynew, rbf(xnew, ynew), level, extent=(-fact*alat_x,fact*alat_x,-fact*alat_y,fact*alat_y), cmap=plt.cm.RdYlBu_r)
        cbar1=plt.colorbar(zt1,ax=ax,orientation=anglerot,shrink=shrin)
        cbar1.set_label(r'$E_{adh} (J/m^2)$', rotation=270, labelpad=20,fontsize=15,family='serif')
        plt.title("PES/MEP for " + self.ctx.element + self.ctx.lattice,fontsize=18,family='serif')
        ax.plot(xi, yi,'.-', c='white', ms=1)
        ax.plot(x1, y1,'.-', c='yellow', ms=2)
        ax.plot(xj, yj,'.-', c='white', ms=1)
        ax.plot(x2, y2,'.-', c='magenta', ms=2)
        ax.axis([-fact*alat_x,fact*alat_x,-fact*alat_y,fact*alat_y])
        plt.xlabel(r"distance ($\AA$)",fontsize=12,family='serif')
        plt.ylabel(r"distance ($\AA$)",fontsize=12,family='serif')
        plt.savefig("mep"+self.ctx.element+self.ctx.lattice+".pdf")
        #plt.draw()
        dx1 = x1-np.roll(x1,1)
        dy1 = y1-np.roll(y1,1)
        dx1[0] = 0.
        dy1[0] = 0.
        tx1 = 0.5*(np.roll(x1, -1)-np.roll(x1, 1))
        ty1 = 0.5*(np.roll(y1, -1)-np.roll(y1, 1))
        dx2 = x2-np.roll(x2,1)
        dy2 = y2-np.roll(y2,1)
        dx2[0] = 0.
        dy2[0] = 0.
        tx2 = 0.5*(np.roll(x2, -1)-np.roll(x2, 1))
        ty2 = 0.5*(np.roll(y2, -1)-np.roll(y2, 1))
        # potential computed as integral of projection of gradV on string tangent
        Vz1 = np.zeros(n1)

        #derivative of the potential
        x1 += delta
        tempValp=rbf(x1,y1)
        x1 -= 2.*delta
        tempValm=rbf(x1,y1)
        dVx1 = 0.5*(tempValp-tempValm)/delta
        x1 += delta
        y1 += delta
        tempValp=rbf(x1,y1)
        y1 -= 2.*delta
        tempValm=rbf(x1,y1)
        y1 += delta
        dVy1 = 0.5*(tempValp-tempValm)/delta

        tforce1= -(tx1*dVx1+ty1*dVy1)
        force1= tforce1/np.sqrt(tx1**2+ty1**2)
        Vz2 = np.zeros(n1)

        #derivative of the potential
        x2 += delta
        tempValp=rbf(x2,y2)
        x2 -= 2.*delta
        tempValm=rbf(x2,y2)
        dVx2 = 0.5*(tempValp-tempValm)/delta
        x2 += delta
        y2 += delta
        tempValp=rbf(x2,y2)
        y2 -= 2.*delta
        tempValm=rbf(x2,y2)
        y2 += delta
        dVy2 = 0.5*(tempValp-tempValm)/delta

        tforce2= -(tx2*dVx2+ty2*dVy2)
        force2= tforce2/np.sqrt(tx2**2+ty2**2)
        with open('output.txt', 'a') as out:
            out.write("Shear strength estimated along MEP(1) from min/max force/area = {0} / {1}  GPa\n".format(str(10.*np.min(force1)),str(10.*np.max(force1))))
            out.write("Shear strength estimated along MEP(2) from min/max force/area = {0} / {1}  GPa\n".format(str(10.*np.min(force2)),str(10.*np.max(force2))))
#        print('Shear strength estimated along MEP(1) from min/max force/area = %.5g / %.5g  GPa' % (10.*np.min(force1),10.*np.max(force1)) )
#        print('Shear strength estimated along MEP(2) from min/max force/area = %.5g / %.5g  GPa' % (10.*np.min(force2),10.*np.max(force2)) )
        for i in range(n1-1):
            Vz1[i+1]=Vz1[i] - 0.5*(tforce1[i]+tforce1[i+1])
            Vz2[i+1]=Vz2[i] - 0.5*(tforce2[i]+tforce2[i+1])
        Vz1-= np.min(Vz1)
        Ve1 = rbf(x1,y1)
        Ve1-= np.min(Ve1)
        lxy1  = np.cumsum(np.sqrt(dx1**2+dy1**2))
        Vz2-= np.min(Vz2)
        Ve2 = rbf(x2,y2)
        Ve2-= np.min(Ve2)
        lxy2  = np.cumsum(np.sqrt(dx2**2+dy2**2))
        fig4 = plt.figure(figsize=(10, 9), dpi=100)
        a1 = fig4.add_subplot(221)
        a1.plot(lxy1,Vz1,'g-',lxy1,Ve1,'b.',ms=3)
        a2 = fig4.add_subplot(222)
        a2.plot(lxy2,Vz2,'g-',lxy2,Ve2,'b.',ms=3)
        a3 = fig4.add_subplot(223)
        a3.plot(lxy1,10.*np.sqrt(dVx1**2+dVy1**2),'g-',lw=0.5)
        a3.plot(lxy1,10.*force1,'.-',color='blue',ms=3)
        a4 = fig4.add_subplot(224)
        a4.plot(lxy2,10.*np.sqrt(dVx2**2+dVy2**2),'g-',lw=0.5)
        a4.plot(lxy2,10.*force2,'.-',color='blue',ms=3)
        plt.savefig("mep-1d"+self.ctx.element+self.ctx.lattice+".pdf")

        with open('output.txt', 'a') as out:
            out.write("\nCALCULATION FINISHED!\n")
#        print "Calculation finished!"



if __name__ == "__main__":
    store_results = run(ShearStrengthWF)
