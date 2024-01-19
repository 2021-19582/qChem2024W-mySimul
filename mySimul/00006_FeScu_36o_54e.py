"""
Generating FCIDUMP for the final Hamiltonian of ASP using pyscf.

Author: Seunghoon Lee, Jan 17, 2022
Edited: Inyoung  Choi, Jan 15, 2024

DOI: 10.1038/NCHEM.2041
36o 54e [Fe4S4(SCH3)4]2- CAS DMRG-CI
FCIDUMP 00006
*NEEDS EDITING*
"""
import sys
import numpy as np
from tools_io import dumpERIs
from pyscf import gto, scf, dft, ao2mo
# FCI: Full Config Integral
# ASP: adiabatic state preparation

#==================================================================
# TODO
#==================================================================
myfilestr = '00006'
mymolstr = 'FeScu'
myactOEstr = '36o_54e'
myorb = 36
myelec = [27, 27]
# Molecule
myatom = '''                                             
 S                  0.04000000   -1.78000000   -1.29000000
 S                 -0.04000000    1.78000000   -1.29000000
 S                  1.78000000   -0.04000000    1.29000000
 S                 -1.78000000    0.04000000    1.29000000
Fe                  0.05000000   -1.37000000    1.01000000
Fe                 -1.38000000    0.05000000   -1.00000000
Fe                 -0.05000000    1.38000000    1.00000000
Fe                  1.37000000   -0.05000000   -1.01000000
 S                  0.24000000    3.30000000    2.14000000
 S                 -0.24000000   -3.29000000    2.14000000
 S                 -3.29000000   -0.24000000   -2.14000000
 S                  3.29000000    0.24000000   -2.14000000
 C                 -3.80000000   -1.84000000   -1.38000000
 H                 -3.91000000   -1.71000000   -0.29000000
 H                 -4.76000000   -2.17000000   -1.81000000
 H                 -3.03000000   -2.60000000   -1.56000000
 C                  3.80000000    1.83000000   -1.38000000
 H                  3.91000000    1.71000000   -0.29000000
 H                  4.76000000    2.16000000   -1.81000000
 H                  3.03000000    2.59000000   -1.55000000
 C                 -1.83000000   -3.80000000    1.38000000
 H                 -2.16000000   -4.76000000    1.81000000
 H                 -2.59000000   -3.03000000    1.55000000
 H                 -1.70000000   -3.91000000    0.29000000
 C                  1.84000000    3.80000000    1.38000000
 H                  2.17000000    4.76000000    1.81000000
 H                  2.60000000    3.03000000    1.56000000
 H                  1.71000000    3.91000000    0.29000000
'''
mycharge = -2
myspin = 10 # NOT SURE!
mysymmetry = False
# SCF
mychkfile =  './output/'+myfilestr+'_hs_bp86_MCSCF_'+mymolstr+'_'+myactOEstr+'.chk'
# genMolden
myNEEDMOLDEN = True
myMOLDENfile =  ''+myfilestr+'_'+mymolstr+'_DFT.molden'
# dumpInt
myact = [] # TODO
myheader = """ &FCI NORB=36,NELEC=54,MS2=0,   
   ORBSYM=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
   ISYM=1,
  &END
 """

myFCIDUMPfile = './output/'+myfilestr+'_MCSCF_'+mymolstr+'_'+myactOEstr+'.FCIDUMP'
#==================================================================
# Molecule
#==================================================================
mol = gto.Mole()                # Mole class: Handle params and attrs for GTO
mol.verbose = 5
mol.atom = myatom

mol.basis = 'tzp-dkh'
mol.charge = mycharge
mol.spin = myspin               # NOT 2S+1 but S_A - S_B
				# curr. dealing Fe_A: 5alpha; Fe_B: 5beta
mol.build()
mol.symmetry = mysymmetry
mol.build()

#==================================================================
# SCF: Self Consistent Field
#==================================================================
mf = scf.sfx2c(scf.RKS(mol))    # RKS: (non-rel) restr. Kohn-Sham
                                # scf.RKS(mol) returns dft.RKS(mol)
                                # sfx2c: spin-free (the scalar part) 
                                #       X2C (eXact-2-component) 
                                #       with 1-electron X-matrix
mf.chkfile = mychkfile
mf.max_cycle = 500
mf.conv_tol = 1.e-4
mf.xc = 'b88,p86' 
mf.scf()			# FAST & ROUGH CONV.

mf2 = scf.newton(mf)            # scf.RHF returns instance of SCF class
                                # newton(mf): Co-iterative augmented hessian (CIAH)
                                #       second order SCF solver
mf2.chkfile = mychkfile
mf2.conv_tol = 1.e-12
mf2.kernel()			# SLOW & PRECISE CONV.

#sys.exit("end of SCF")
#=================================================================
# Generate molden
#=================================================================
mo = mf2.mo_coeff

if myNEEDMOLDEN :
    from pyscf import tools
    tools.molden.from_mo(mol, myMOLDENfile, mo)
    sys.exit("end of gen molden")
#==================================================================
# Dump integrals
#==================================================================
mo = mf2.mo_coeff               
norb = myorb
nelec = myelec 		        # [ # of alpha e, # of beta e ]


from pyscf import mcscf         # MCSCF: Multi-configuration self-consistent field


mc = mcscf.CASCI(mf, norb, nelec)
                                # CASCI: Complete Active Space config. integral
                                #       wave ftn is L-comb of Slater-det.s
                                #       expansion coeff solved in variational procedure
                                #   Note) fixed oribitals!
                     
mc.mo_coeff = mo                # [Fe2S2(SCH3)2]2-
act_idx = myact
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




header = myheader
dumpERIs(myFCIDUMPfile, header, int1e=h1e, int2e=g2e, ecore=ecore)       
                                # ERI: electron repulsion integral
                                # Final Hamiltonian
                                
