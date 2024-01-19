
"""
Dump integrals from chkfile using pyscf.

Author: Seunghoon Lee, Jan 17, 2022
Edited: Inyoung  Choi, Jan 16, 2024

"""

import numpy as np
from tools_io import dumpERIs
from pyscf import gto, scf, dft, ao2mo
# FCI: Full Config Integral
# ASP: adiabatic state preparation

#==================================================================
# TODO 
#==================================================================
mychkfile = './output/00003_hs_bp86_MCSCF_FeSdi_OX_20o_30e.chk'
norb = 20
nelec = [15, 15]                # [ # of a, # of b ]
header =""" &FCI NORB=20,NELEC=30,MS2=0,
  ORBSYM=1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
  ISYM=1,
 &END
"""
myFCIDUMPfile = './output/00003_TESTdumpInt_MCSCF_FeSdi_OX_20o_30e.FCIDUMP'


act_idx = [] # MO indexes of active space
###
act_s_3p = [21, 24, 32, 33]
act_s2_3p = [40, 43, 44]
act_s5_3p = [39, 41, 42]
act_cp_op = act_s_3p + act_s2_3p + act_s5_3p

			        # p-orbitals of S...
act_d = [89, 90, 91, 92, 93, 94, 95, 96, 97, 98]
                                # d-orbitals of Fe
act_idx = act_cp_op + act_d                             
###




#==================================================================
# Dump integrals from chkfile
#==================================================================
from pyscf import lib

mol = lib.chkfile.load_mol(mychkfile)
mo = lib.chkfile.load(mychkfile, 'scf/mo_coeff')


from pyscf import mcscf         # MCSCF: Multi-configuration self-consistent field
mc = mcscf.CASCI(scf.sfx2c(scf.RKS(mol)), norb, nelec)

                                # CASCI: Complete Active Space config. integral
                                #       wave ftn is L-comb of Slater-det.s
                                #       expansion coeff solved in variational procedure
                                #   Note) fixed oribitals!
                     
mc.mo_coeff = mo                # [Fe2S2(SCH3)2]2-

assert len(act_idx) == norb     # norb: # of orbitals

mo = mc.sort_mo(mo_coeff = mo,caslst=act_idx)        # pushes empty spaces in list (...)



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
dumpERIs(myFCIDUMPfile, header, int1e=h1e, int2e=g2e, ecore=ecore)       
                                # ERI: electron repulsion integral
                                # Final Hamiltonian
