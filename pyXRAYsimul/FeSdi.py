#    PyXray: a library for ab-initio X-ray spectrum simulation
#    Copyright (C) 2023  Seunghoon Lee <seunghoonlee89@gmail.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#

from pyscf import gto, scf, symm, ao2mo, cc, mcscf, tools
from pyscf.data import nist
import h5py
import numpy as np

class ActiveSpaceModel():
    def __init__(self, model):
        model_list = ['FeSdi_OX_12o14e_lunoloc', 'FeSdi_OX_20o30e_lunoloc', 'FeSdi_OX_20!o30e_lunoloc', 'FeSdi_OX_20!!o30e_lunoloc']
        assert model in model_list
        self.model = model

        self.mol = None 
        self.mf = None 
        self.mc = None 
        self.mo_coeff = None 
        self.mo_occ = None 
        self.norb = None 
        self.n_elec = None 
        self.twos = None 
        self.h1e = None 
        self.g2e = None
        self.ecore = None
        self.hso = None
        self.hr = None
        self.HARTREE2EV = nist.HARTREE2EV
        self.LIGHT_SPEED = nist.LIGHT_SPEED   # in a.u.
        self.n_core = None 
        self.n_active = None 
        self.n_inactive = 0 
        self.n_external = 0 
        self.init_sys = False 

    def gen_mol(self, verbose=5):
        if 'FeSdi' in self.model:
            atom = """  Fe                 5.22000000    1.05000000   -7.95000000
            S                  3.86000000   -0.28000000   -9.06000000
            S                  5.00000000    0.95000000   -5.660000
            S                  4.77000000    3.18000000   -8.740000
            S                  7.23000000    0.28000000   -8.380000
            Fe                 5.88000000   -1.05000000   -9.490000
            S                  6.10000000   -0.95000000  -11.790000
            S                  6.33000000   -3.18000000   -8.710000
            C                  6.00000000    4.34000000   -8.170000
            H                  6.46000000    4.81000000   -9.010000
            H                  5.53000000    5.08000000   -7.550000
            H                  6.74000000    3.82000000   -7.600000
            C                  3.33000000    1.31000000   -5.180000
            H                  2.71000000    0.46000000   -5.370000
            H                  3.30000000    1.54000000   -4.130000
            H                  2.97000000    2.15000000   -5.730000
            C                  5.10000000   -4.34000000   -9.280000
            H                  5.56000000   -5.05000000   -9.930000
            H                  4.67000000   -4.84000000   -8.440000
            H                  4.34000000   -3.81000000   -9.810000
            C                  7.77000000   -1.31000000  -12.270000
            H                  7.84000000   -1.35000000  -13.340000
            H                  8.42000000   -0.54000000  -11.900000
            H                  8.06000000   -2.25000000  -11.860000
            """
            charge = -2
            twos = 10    # nb - na 
        else:
            assert False 

        mpg = 'c1'  # point group: d2h or c1

        self.mol = gto.M(atom=atom, symmetry=mpg, basis= 'tzp-dkh',
                         spin=twos, charge=charge, verbose=verbose)
        return self.mol

    def do_mf(self, mol, chkfile=None):
        from pyscf import scf
        mf = scf.sfx2c(scf.UKS(mol))
        if chkfile is not None:
            mf.chkfile = '%s_uks.h5' % self.model
        mf.max_cycle = 100
        mf.conv_tol=1.e-9
        mf.level_shift = 0.3
        mf.xc = 'b88,p86' 
        mf.kernel()
        
        mf = scf.newton(mf)
        mf.conv_tol = 1.e-12
        mf.kernel()
        return mf

    def init_model(self, chkfile, MPI=None, method='casci', local=False, dbg=False):
        self.init_sys = True
        assert method in ['casci', 'mrcis']
        model = self.model

        # load dumpfile 
        self.mf = scf.ROHF(self.mol)
        if '.h5' == chkfile[-3:] and '_lunoloc' in model:
            f = h5py.File(chkfile,'r')
            self.mo_coeff = np.array(f['luno']['mo_coeff'])
            f.close()
        else:
            assert False
        self.mf.mo_coeff = self.mo_coeff

        # define CAS model
        if 'FeSdi_OX' in model:
            if '_lunoloc' in model:
                if '12o14e' in model:
                    act0 = [78, 84] # bridging s 3pz
                    act1 = [89, 90, 91, 92, 93, 94, 95, 96, 97, 98] # fe 3d
                    idx = act0+act1
                    na = 7
                    nb = 7
                    act_3p = act0
                    act_3d = act1
                    self.n_core = len(act0) 
                    self.n_active = len(act1)
                elif '20o30e' in model:
                    act0 = [81, 82, 85, 86]+[75, 76, 77, 78, 79, 83]  # s 3p
                    act1 = [87, 88, 91, 92, 93, 94, 95, 96, 97, 98] # fe 3d
                    idx = act0+act1
                    na = 7
                    nb = 7
                    act_3p = act0
                    act_3d = act1
                    self.n_core = len(act0)
                    self.n_active = len(act1)          
                elif '20!o30e' in model:
                    act0 = [79, 80, 81, 82, 83, 84, 85, 86, 87, 88]  # s 3p
                    act1 = [89, 90, 91, 92, 93, 94, 95, 96, 97, 98] # fe 3d
                    idx = act0+act1
                    na = 7
                    nb = 7
                    act_3p = act0
                    act_3d = act1
                    self.n_core = len(act0)
                    self.n_active = len(act1)
                elif '20!!o30e' in model:
                    act0 = [73, 75, 76, 77, 78, 79, 81, 82, 83, 84] # s 3p
                    act1 = [89, 90, 91, 92, 93, 94, 95, 96, 97, 98] # fe 3d
                    idx = act0+act1
                    na = 7
                    nb = 7
                    act_3p = act0
                    act_3d = act1
                    self.n_core = len(act0)
                    self.n_active = len(act1)
                else:
                    assert False
            else:
                assert False
        else:
            assert False

        if method == 'mrcis':
            if 'feIII_' in model and '_lunoloc' in model:
                sigmap = [34,35,36,37] # fe-cl sigma 
                inact_idx = sigmap
                virt_idx = [] 
            elif 'feII_' in model and '_lunoloc' in model:
                sigmap = [34,35,36,37] # fe-cl sigma 
                inact_idx = sigmap
                virt_idx = [] 
            else:
                assert False

            self.n_inactive = len(inact_idx)
            self.n_external = len(virt_idx) 
            idx = act_2p + inact_idx + virt_idx + act_3d
            na += len(inact_idx)
            nb += len(inact_idx)

        self.idx = idx      
        self.n_elec = (na, nb) 
        self.norb = len(idx) 
        
        self.mc = mcscf.CASCI(self.mf, self.norb, self.n_elec)
        self.mc.mo_coeff = self.mo_coeff
        self.mo_coeff = self.mc.sort_mo(idx)
        if local:
            if method == 'casci':
                nc = self.mc.ncore
                norb = self.mc.ncas
                clmo = self.mo_coeff[:, :nc].copy() 
                almo = self.mo_coeff[:, nc:nc+norb].copy() 
                vlmo = self.mo_coeff[:, nc+norb:].copy() 
            elif method == 'mrcis':
                nc = self.mc.ncore
                nic = self.n_core
                ni = self.n_inactive
                na = self.n_active
                ne = self.n_external
                norb = self.mc.ncas
                assert ni + na + ne + 3 == norb
                clmo = self.mo_coeff[:, :nc+nic+ni].copy() 
                almo = self.mo_coeff[:, nc+nic+ni:nc+nic+ni+na].copy() 
                vlmo = self.mo_coeff[:, nc+nic+ni+na:].copy() 

            if dbg:
                print('before loc')
                from pyxray.utils.addons import lowdinPop
                lowdinPop(self.mol, almo)

            from pyscf import lo
            almo = lo.PM(self.mol, almo).kernel()
            lmo = np.hstack((clmo, almo, vlmo)).copy()
            self.mo_coeff = lmo 
            self.mc.mo_coeff = self.mo_coeff

            if dbg:
                print('after loc')
                from pyxray.utils.addons import lowdinPop
                lowdinPop(self.mol, almo)
        else:
            self.mc.mo_coeff = self.mo_coeff

            if dbg:
                print('Lowdin Pop: Active Orbitals')
                nc = self.mc.ncore
                norb = self.mc.ncas
                almo = self.mo_coeff[:, nc:nc+norb]
                from pyxray.utils.addons import lowdinPop
                lowdinPop(self.mol, almo)

        # check active space model
        if dbg:
            from pyscf.tools import molden
            nc  = self.mc.ncore
            nca = self.mc.ncore + self.norb
            mocas = self.mo_coeff[:,nc:nca]
            with open('%s.molden' % model, 'w') as f1:
                molden.header(self.mol, f1)
                molden.orbital_coeff(self.mol, f1, mocas)

    def gen_ham(self, tol=1e-9):
        # Integrals for Spin-Free ab-initio CAS Hamiltonian
        assert self.init_sys
        h1e, ecore = self.mc.get_h1eff()
        h1e[np.abs(h1e) < tol] = 0
        g2e = self.mc.get_h2eff()
        g2e = ao2mo.restore(1, g2e, self.norb)
        g2e[np.abs(g2e) < tol] = 0
        return h1e, g2e, ecore

    def gen_hso(self, somf=True, amfi=True):
        # Integrals for Breit-Pauli SOC Hamiltonian
        assert self.init_sys
        if somf:
            ncore = self.mc.ncore
            nmo = self.mf.mo_coeff.shape[0]
            na, nb = self.n_elec
            occa = np.array([1 if i < ncore + na else 0 for i in range(nmo)])
            occb = np.array([1 if i < ncore + nb else 0 for i in range(nmo)]) 
            self.mo_occ = occa + occb
            def gen_mf_dmao(mo_coeff, mo_occ):
                mocc = mo_coeff[:,mo_occ>0]
                return np.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)
    
            from pyxray.utils.integral_helper import compute_hso_mo  
            from pyxray.utils.integral_helper import compute_hso_ao  
            dmao = gen_mf_dmao(self.mo_coeff, self.mo_occ) 
            mocas = self.mc.mo_coeff[:,ncore:ncore+self.norb]
            hso = compute_hso_mo(self.mol, dmao, mocas, amfi=amfi)
            return hso, None
        else:
            ncore = self.mc.ncore
            mocas = self.mc.mo_coeff[:, ncore:ncore+self.norb]
            from pyxray.utils.integral_helper import compute_bpso_mo  
            hso1e, hso2e = compute_bpso_mo(self.mol, mocas)
            return hso1e, hso2e 

    def gen_hr(self):
        # Integrals for dipole operator 
        assert self.init_sys
        hrao = self.mol.intor('int1e_r')
        ncore = self.mc.ncore
        mocas = self.mc.mo_coeff[:, ncore:ncore+self.norb]
        hr = np.einsum('rij,ip,jq->rpq', hrao, mocas, mocas)
        return hr 

    def do_fci(self, h1e, g2e, norb=None, n_elec=None, nroots=1):
        assert self.init_sys == True
        if norb is None:
            norb = self.norb
        if n_elec is None:
            n_elec = self.n_elec
        from pyscf import fci
        e, fcivec = fci.direct_spin1.kernel(h1e, g2e, norb, n_elec, nroots=nroots,
                                            max_space=1000, max_cycle=1000)
        return e, fcivec

