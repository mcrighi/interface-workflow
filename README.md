# AiiDA workflow for homogeneous interface

Official page of the AiiDA workflow able to calculate adhesion energy and ideal interfecial shear strength of homogeneous solid interfaces by means of first principles calculations.

## Getting Started

To run the workflow it is necessary to install AiiDA v0.12.0. Further details on AiiDA installation can be found [here](http://aiida-core.readthedocs.io/en/stable/installation/index.html).

## Run calculation

Once AiiDA is properly installed, it is possible to run the workflow. First, it is necessary to modify the input file (further details for the input parameters can be foud in the [input.in](input.in) page). Then, it is possible to run the workflow with the following command:

      nohup verdi run work_shearstrength_2_0.py > output.out &

## Calculation outputs

The workflow generates an output file with the most relevant computed quantities. These quantities are also stored inside the AiiDA database. Here it is a list of the most important quantities generated nd stored by the workflow:

* the converged kinetic energy cutoff for the wavefunctions
* the optimal k-point density
* the optimized lattice parameter
* the surface energy for the selected surface
* the two dimensional Potential Energy Surface (PES) and the Minimum Energy Path (MEP) for the selected surface
* the perpendicular potential profile at the minimum and maximum values of the lateral PES
* the electronic charge displacement at the minimum and maximum positions of the PES
* the adhesion energy
* the ideal interfacial shear strength along the MEP and in two high symmetry directions orthogonal to each other

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
