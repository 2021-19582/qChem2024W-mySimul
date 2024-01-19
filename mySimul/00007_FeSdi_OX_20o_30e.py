"""
Generating FCIDUMP for the final Hamiltonian of ASP using pyscf.

Author: Seunghoon Lee, Jan 17, 2022
Edited: Inyoung  Choi, Jan 15, 2024

Final ACT for FeSdi_OX
FCIDUMP 00007
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
myfilestr = '00007'
mymolstr = 'FeSdi_OX'
myactOEstr = '20o_30e'
myorb = 20
myelec = [15, 15]
# Molecule
myatom = """
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
""" 


mycharge = -2 
myspin = 10 
mysymmetry = False 
# SCF
mychkfile =  './output/'+myfilestr+'_hs_bp86_MCSCF_'+mymolstr+'_'+myactOEstr+'.chk' 
# genMolden
myNEEDMOLDEN = False
myMOLDENfile =  ''+myfilestr+'_'+mymolstr+'_DFT.molden' 
# dumpInt
myact = [] 
#mySn = [61, 62, 
mySn = [81, 82, 85, 86]

#mySb = [57, 58, 59, 60
mySb = [75, 76, 77, 78, 79, 83] 
myS = mySn + mySb
myFe = [87, 88, 91, 92, 93, 94, 95, 96, 97, 98] # 10 electrons
myact = myS + myFe

myheader = """ &FCI NORB=20,NELEC=30,MS2=0,   
   ORBSYM=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
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
                                
