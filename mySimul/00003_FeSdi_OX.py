
"""
Generating FCIDUMP for the final Hamiltonian of ASP using pyscf.

Author: Seunghoon Lee, Jan 17, 2022
Edited: Inyoung  Choi, Jan 11, 2024

DOI: 10.1038/NCHEM.2041
20o 30e [Fe2S2(SCH3)4]2- CAS DMRG-CI
FCIDUMP 00003

"""

import numpy as np
from tools_io import dumpERIs
from pyscf import gto, scf, dft, ao2mo
# FCI: Full Config Integral
# ASP: adiabatic state preparation

#==================================================================
# Molecule
#==================================================================
mol = gto.Mole()                # Mole class: Handle params and attrs for GTO
mol.verbose = 5
                                # mol.atom: coords of atoms [Fe2S2(SCH3)4]2-
mol.atom = '''
 Fe                 5.22000000    1.05000000   -7.95000000
 S                  3.86000000   -0.28000000   -9.06000000
 S                  5.00000000    0.95000000   -5.66000000
 S                  4.77000000    3.18000000   -8.74000000
 S                  7.23000000    0.28000000   -8.38000000
 Fe                 5.88000000   -1.05000000   -9.49000000
 S                  6.10000000   -0.95000000  -11.79000000
 S                  6.33000000   -3.18000000   -8.71000000
 C                  6.00000000    4.34000000   -8.17000000
 H                  6.46000000    4.81000000   -9.01000000
 H                  5.53000000    5.08000000   -7.55000000
 H                  6.74000000    3.82000000   -7.60000000
 C                  3.33000000    1.31000000   -5.18000000
 H                  2.71000000    0.46000000   -5.37000000
 H                  3.30000000    1.54000000   -4.13000000
 H                  2.97000000    2.15000000   -5.73000000
 C                  5.10000000   -4.34000000   -9.28000000
 H                  5.56000000   -5.05000000   -9.93000000
 H                  4.67000000   -4.84000000   -8.44000000
 H                  4.34000000   -3.81000000   -9.81000000
 C                  7.77000000   -1.31000000  -12.27000000
 H                  7.84000000   -1.35000000  -13.34000000
 H                  8.42000000   -0.54000000  -11.90000000
 H                  8.06000000   -2.25000000  -11.86000000
'''
mol.basis = 'tzp-dkh'
mol.charge = -2			# FeS_OX
mol.spin = 10 			# NOT 2S+1 but S_A - S_B
				# curr. dealing Fe_A: 5alpha; Fe_B: 5beta
mol.build()
mol.symmetry = False
mol.build()

#==================================================================
# SCF: Self Consistent Field
#==================================================================
mf = scf.sfx2c(scf.RKS(mol))    # RKS: (non-rel) restr. Kohn-Sham
                                # scf.RKS(mol) returns dft.RKS(mol)
                                # sfx2c: spin-free (the scalar part) 
                                #       X2C (eXact-2-component) 
                                #       with 1-electron X-matrix
mf.chkfile =    './output/hs_bp86_MCSCF_FeSdi_OX_20o_30e_00003.chk'
mf.max_cycle = 500
mf.conv_tol = 1.e-4
mf.xc = 'b88,p86' 
mf.scf()			# FAST & ROUGH CONV.

mf2 = scf.newton(mf)            # scf.RHF returns instance of SCF class
                                # newton(mf): Co-iterative augmented hessian (CIAH)
                                #       second order SCF solver
mf2.chkfile =   './output/hs_bp86_MCSCF_FeSdi_OX_20o_30e_00003.chk'
mf2.conv_tol = 1.e-12
mf2.kernel()			# SLOW & PRECISE CONV.

#==================================================================
# Dump integrals
#==================================================================
mo = mf2.mo_coeff               
norb = 20
nelec = [15, 15] 		# [ # of alpha e, # of beta e ]


from pyscf import mcscf         # MCSCF: Multi-configuration self-consistent field


mc = mcscf.CASCI(mf, norb, nelec)
                                # CASCI: Complete Active Space config. integral
                                #       wave ftn is L-comb of Slater-det.s
                                #       expansion coeff solved in variational procedure
                                #   Note) fixed oribitals!
                     
mc.mo_coeff = mo                # [Fe2S2(SCH3)2]2-
act_s_3p = [21, 24, 32, 33]
act_s2_3p = [40, 43, 44]
act_s5_3p = [39, 41, 42]
act_cp_op = act_s_3p + act_s2_3p + act_s5_3p

			        # p-orbitals of S...
act_d = [89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
                                # d-orbitals of Fe
act_idx = act_cp_op + act_d
assert len(act_idx) == norb     # norb: # of orbitals
mo = mc.sort_mo(act_idx)        # pushes empty spaces in list (...)
mc.mo_coeff = mo
#from pyscf import tools
#tools.molden.from_mo(mol, 'fe2s2_actonly.molden', mo[:,86:98])

h1e, ecore = mc.get_h1eff()     # get_h1eff(): CAS space one-electron hamiltonian
                                # h1e: effective one-electron hamiltonian 
                                #       defined in CAS space
                                # ecore: electronic energy from core
g2e = mc.get_h2eff()            # get_h2eff(): active space two-particle Hamiltonian
g2e = ao2mo.restore(1, g2e, norb)
                                # g2e: converted the 2e intgrl (in Chemist's notation)
                                #       between different
                                #       level of permutation symmetry
header =""" &FCI NORB=20,NELEC=30,MS2=0,
  ORBSYM=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  ISYM=1,
 &END
"""
dumpERIs('./output/00003_MCSCF_FeSdi_OX_20o_30e.FCIDUMP', header, int1e=h1e, int2e=g2e, ecore=ecore)       
                                # ERI: electron repulsion integral
                                # Final Hamiltonian
