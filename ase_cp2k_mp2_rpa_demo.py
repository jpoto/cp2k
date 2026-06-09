#!/usr/bin/env python3
"""
Demonstration of using ASE with CP2K for MP2/RPA calculations.

This script shows how to set up and run MP2/RPA calculations using ASE's CP2K interface.
Note: For MP2/RPA, we need to use the regular CP2K interface (not shell mode).
"""

import os
import sys

# Add the virtual environment's site-packages to Python path
venv_path = os.path.expanduser('~/venvs/ase/lib/python3.12/site-packages')
if os.path.exists(venv_path):
    sys.path.insert(0, venv_path)

import numpy as np
from ase import Atoms
from ase.calculators.cp2k import CP2K
from ase.optimize import BFGS

def create_h2o_molecule():
    """Create a water molecule for demonstration."""
    # H2O molecule: O at (0,0,0), H at (0.958, 0.0, 0.0), H at (0.0, 0.958, 0.0)
    # Bond angle ~104.5 degrees
    molecules = Atoms('H2O', 
                      positions=[[0.0, 0.0, 0.0],
                                [0.958, 0.0, 0.0],
                                [0.0, 0.958, 0.0]])
    molecules.center(vacuum=5.0)  # Add vacuum space
    return molecules

def setup_cp2k_mp2_calculation():
    """Set up CP2K calculator for MP2 calculation."""
    molecules = create_h2o_molecule()
    
    # CP2K MP2 calculation parameters
    calc = CP2K(
        # Basic setup
        label='h2o_mp2',
        basis_set='DZVP-MOLOPT-SR-GTH',
        potential='GTH-PBE',
        cutoff=400,  # in Ry
        rel_cutoff=60,
        
        # DFT setup (required for MP2)
        xs_section_params={
            'METHOD': 'PBE',
            'XC_FUNCTIONAL': {'_': 'PBE'}
        },
        
        # MP2 setup
        dft_section_params={
            'MP2': {
                'METHOD': 'GPW',
                'RI_MP2': True,
                'MEMORY': {'EPS_STORAGE_SCALING': 0.1},
                'SCF': {
                    'MAX_SCF': 50,
                    'EPS_SCF': 1e-6,
                    'SCF_GUESS': 'ATOMIC'
                }
            }
        },
        
        # Output settings
        print_level='MEDIUM',
        
        # Parallelization
        mpi_settings={
            'RUN_TYPE': 'ENERGY_FORCE'
        }
    )
    
    return molecules, calc

def setup_cp2k_rpa_calculation():
    """Set up CP2K calculator for RPA calculation."""
    molecules = create_h2o_molecule()
    
    # CP2K RPA calculation parameters
    calc = CP2K(
        # Basic setup
        label='h2o_rpa',
        basis_set='DZVP-MOLOPT-SR-GTH',
        potential='GTH-PBE',
        cutoff=400,  # in Ry
        rel_cutoff=60,
        
        # DFT setup (required for RPA)
        xs_section_params={
            'METHOD': 'PBE',
            'XC_FUNCTIONAL': {'_': 'PBE'}
        },
        
        # RPA setup
        dft_section_params={
            'RPA': {
                'METHOD': 'GPW',
                'RI_RPA': True,
                'MEMORY': {'EPS_STORAGE_SCALING': 0.1},
                'SCF': {
                    'MAX_SCF': 50,
                    'EPS_SCF': 1e-6,
                    'SCF_GUESS': 'ATOMIC'
                }
            }
        },
        
        # Output settings
        print_level='MEDIUM',
        
        # Parallelization
        mpi_settings={
            'RUN_TYPE': 'ENERGY_FORCE'
        }
    )
    
    return molecules, calc

def run_mp2_calculation():
    """Run MP2 calculation using ASE + CP2K."""
    print("Setting up MP2 calculation...")
    molecules, calc = setup_cp2k_mp2_calculation()
    
    # Attach calculator
    molecules.calc = calc
    
    print("Running MP2 calculation...")
    try:
        # Calculate energy
        energy = molecules.get_potential_energy()
        print(f"MP2 Energy: {energy} eV")
        
        # Calculate forces
        forces = molecules.get_forces()
        print(f"Forces shape: {forces.shape}")
        print("Max force magnitude:", np.max(np.linalg.norm(forces, axis=1)))
        
        return molecules, calc, energy
        
    except Exception as e:
        print(f"MP2 calculation failed: {e}")
        return None, None, None

def run_rpa_calculation():
    """Run RPA calculation using ASE + CP2K."""
    print("Setting up RPA calculation...")
    molecules, calc = setup_cp2k_rpa_calculation()
    
    # Attach calculator
    molecules.calc = calc
    
    print("Running RPA calculation...")
    try:
        # Calculate energy
        energy = molecules.get_potential_energy()
        print(f"RPA Energy: {energy} eV")
        
        # Calculate forces
        forces = molecules.get_forces()
        print(f"Forces shape: {forces.shape}")
        print("Max force magnitude:", np.max(np.linalg.norm(forces, axis=1)))
        
        return molecules, calc, energy
        
    except Exception as e:
        print(f"RPA calculation failed: {e}")
        return None, None, None

def create_mp2_input_file():
    """Create a CP2K input file for MP2 calculation."""
    mp2_input = """&GLOBAL
  PROJECT h2o_mp2
  RUN_TYPE ENERGY_FORCE
  PRINT_LEVEL MEDIUM
&

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME GTH_POTENTIALS
    
    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &
    
    &SCF
      MAX_SCF 50
      EPS_SCF 1.0E-6
      SCF_GUESS ATOMIC
      &OT
        MINIMIZER DIIS
        PRECONDITIONER FULL_SINGLE_INVERSE
      &
      &OUTER_SCF
        MAX_SCF 10
        EPS_SCF 1.0E-6
      &
    &
    
    &XC
      &XC_FUNCTIONAL PBE
      &
      &VDW_POTENTIAL
        DISPERSION_FUNCTIONAL PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3
          PARAMETER_FILE_NAME dftd3.dat
          REFERENCE_FUNCTIONAL PBE
        &
      &
    &
    
    &MP2
      METHOD GPW
      RI_MP2 TRUE
      MEMORY
        EPS_STORAGE_SCALING 0.1
      &
    &
  &
  
  &SUBSYS
    &CELL
      ABC 10.0 10.0 10.0
      PERIODIC NONE
    &
    &COORD
      O    0.00000000    0.00000000    0.00000000
      H    0.95800000    0.00000000    0.00000000
      H    0.00000000    0.95800000    0.00000000
    &
    &KIND O
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE
    &
    &KIND H
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE
    &
  &
&
"""
    
    with open('h2o_mp2.inp', 'w') as f:
        f.write(mp2_input)
    
    print("Created MP2 input file: h2o_mp2.inp")

def create_rpa_input_file():
    """Create a CP2K input file for RPA calculation."""
    rpa_input = """&GLOBAL
  PROJECT h2o_rpa
  RUN_TYPE ENERGY_FORCE
  PRINT_LEVEL MEDIUM
&

&FORCE_EVAL
  METHOD Quickstep
  &DFT
    BASIS_SET_FILE_NAME BASIS_MOLOPT
    POTENTIAL_FILE_NAME GTH_POTENTIALS
    
    &QS
      METHOD GPW
      EPS_DEFAULT 1.0E-12
    &
    
    &SCF
      MAX_SCF 50
      EPS_SCF 1.0E-6
      SCF_GUESS ATOMIC
      &OT
        MINIMIZER DIIS
        PRECONDITIONER FULL_SINGLE_INVERSE
      &
      &OUTER_SCF
        MAX_SCF 10
        EPS_SCF 1.0E-6
      &
    &
    
    &XC
      &XC_FUNCTIONAL PBE
      &
      &VDW_POTENTIAL
        DISPERSION_FUNCTIONAL PAIR_POTENTIAL
        &PAIR_POTENTIAL
          TYPE DFTD3
          PARAMETER_FILE_NAME dftd3.dat
          REFERENCE_FUNCTIONAL PBE
        &
      &
    &
    
    &RPA
      METHOD GPW
      RI_RPA TRUE
      MEMORY
        EPS_STORAGE_SCALING 0.1
      &
      &SCF
        MAX_SCF 20
        EPS_SCF 1.0E-5
      &
    &
  &
  
  &SUBSYS
    &CELL
      ABC 10.0 10.0 10.0
      PERIODIC NONE
    &
    &COORD
      O    0.00000000    0.00000000    0.00000000
      H    0.95800000    0.00000000    0.00000000
      H    0.00000000    0.95800000    0.00000000
    &
    &KIND O
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE
    &
    &KIND H
      BASIS_SET DZVP-MOLOPT-SR-GTH
      POTENTIAL GTH-PBE
    &
  &
&
"""
    
    with open('h2o_rpa.inp', 'w') as f:
        f.write(rpa_input)
    
    print("Created RPA input file: h2o_rpa.inp")

def main():
    """Main demonstration function."""
    print("ASE + CP2K MP2/RPA Demonstration")
    print("=" * 40)
    
    # Create input files first (these will work with regular CP2K)
    print("\n1. Creating CP2K input files for MP2 and RPA:")
    create_mp2_input_file()
    create_rpa_input_file()
    
    # Set CP2K environment variables
    os.environ['CP2K_DATA_DIR'] = '/workspace/data'
    
    # Try to run calculations using ASE interface
    print("\n2. Attempting MP2 calculation with ASE:")
    try:
        molecules, calc, energy = run_mp2_calculation()
        if energy is not None:
            print(f"✓ MP2 calculation successful! Energy: {energy} eV")
    except Exception as e:
        print(f"✗ MP2 calculation with ASE failed: {e}")
        print("   This is expected - use the input files with regular CP2K")
    
    print("\n3. Attempting RPA calculation with ASE:")
    try:
        molecules, calc, energy = run_rpa_calculation()
        if energy is not None:
            print(f"✓ RPA calculation successful! Energy: {energy} eV")
    except Exception as e:
        print(f"✗ RPA calculation with ASE failed: {e}")
        print("   This is expected - use the input files with regular CP2K")
    
    print("\n4. How to run the calculations:")
    print("   For MP2:  source /workspace/install/cp2k_env && /workspace/install/bin/cp2k.psmp h2o_mp2.inp")
    print("   For RPA:  source /workspace/install/cp2k_env && /workspace/install/bin/cp2k.psmp h2o_rpa.inp")
    
    print("\n5. Key points:")
    print("   - Shell mode (-s) doesn't support MP2/RPA due to architectural limitations")
    print("   - Use regular CP2K input files for MP2/RPA calculations")
    print("   - ASE can generate input files but may not run MP2/RPA directly")
    print("   - The created input files (h2o_mp2.inp, h2o_rpa.inp) are ready to use")

if __name__ == "__main__":
    main()