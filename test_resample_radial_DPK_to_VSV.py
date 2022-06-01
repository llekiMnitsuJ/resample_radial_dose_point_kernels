# -*- coding: utf-8 -*-
"""
Created on Tue May 31 17:30:21 2022

@author: Justin Mikell, mikell@wustlDOTedu

INSTRUCTIONS: update the variable gravesDir to point to the directory 
containing the unzipped "*.processed.csv" files from Graves et al. 

Then you can run this test with pytest. 
I run it from the spyder console via !pytest. 

REFERENCES:
    1) Bolch WE, Bouchet LG, Robertson JS, Wessels BW, Siegel JA, Howell RW, Erdi AK, Aydogan B, Costes S, Watson EE, Brill AB, Charkes ND, Fisher DR, Hays MT, Thomas SR. MIRD pamphlet No. 17: the dosimetry of nonuniform activity distributions--radionuclide S values at the voxel level. Medical Internal Radiation Dose Committee. J Nucl Med. 1999 Jan;40(1):11S-36S. PMID: 9935083.
    2) Graves, S.A., Flynn, R.T. and Hyer, D.E. (2019), Dose point kernels for 2,174 radionuclides. Med. Phys., 46: 5284-5293. https://doi.org/10.1002/mp.13789
    2a) supplemental data from Graves et al https://zenodo.org/record/2564036
    3) Wilderman SJ, Dewaraja YK. Method for Fast CT/SPECT-Based 3D Monte Carlo Absorbed Dose Computations in Internal Emitter Therapy. IEEE Trans Nucl Sci. 2007 Feb 17;54(1):146-151. doi: 10.1109/TNS.2006.889164. PMID: 20305792; PMCID: PMC2841294.

    
USE at your own risk. This test file provides some quanitative values in terms of agreement 
with published data for the monte carlo integration/resampling of the radial dose point kernels. 
The relative agreement factors in this test file were tuned. 
    
"""



gravesDir=r'C:\Users\mikell\OneDrive - Washington University in St. Louis\research\MICH\20190213_Graves_DosePointKernels_v1.0'

from resample_radial_DPK_to_VSV import *
import numpy as np
import pytest

filemap = {}
filemap['P32'] = ["32P_14.26D_beta"]
filemap['Sr89'] = ["89SR_50.53D_beta","89SR_50.563D_electron","89SR_50.563D_gamma"]
filemap['Y90'] = ["90Y_64.10H_beta","90Y_64.00H_electron","90Y_64.00H_gamma"]
filemap['Tc99m'] = ["99TC_6.0072H_electron", "99TC_6.0072H_gamma"]
filemap['I131'] =  ["131I_8.02070D_beta", "131I_8.0252D_electron", "131I_8.0252D_gamma"]

for myKey in filemap:
    mylist = filemap[myKey]
    for i in np.arange(len(mylist)):
        mylist[i] = gravesDir + "\\" + mylist[i] + ".io_processed.csv"
    filemap[myKey] = mylist
        


def test_resample_against_MIRD17_6mm_source_voxel():
    """
    MIRD 17 data taken from Bolch et al JNM 1999 tables. 
    
    'P-32' mGy/MBq/s
    (0,0,0) = 3.19E-1
    (5,5,5) = 7.16E-8
    
    'Sr-89'
    (0,0,0) = 2.85E-1
    (5,5,5) = 4.94E-8
    
    'Y-90'
    (0,0,0) = 3.46E-1
    (5,5,5) = 1.39E-7
    
    'Tc-99m'
    (0,0,0) = 1.2E-2
    (5,5,5) = 2.33E-6

    'I-131'
    (0,0,0) = 1.29E-1
    (5,5,5) = 6.1E-6

    """
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['P32'], i=0,j=0,k=0,dx_mm=6.,dy_mm=6.,dz_mm=6.)    
    mird17val = 3.19E-1
    assert val == pytest.approx(mird17val, 0.02)
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['P32'], i=5,j=5,k=5,dx_mm=6.,dy_mm=6.,dz_mm=6.)    
    mird17val = 7.16E-8
    assert val == pytest.approx(mird17val, 0.5) #large acceptance in brems tail!
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['Sr89'], i=0,j=0,k=0,dx_mm=6.,dy_mm=6.,dz_mm=6.)    
    mird17val = 2.85E-1
    assert val == pytest.approx(mird17val, 0.02)
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['Sr89'], i=5,j=5,k=5,dx_mm=6.,dy_mm=6.,dz_mm=6.)    
    mird17val = 4.94E-8
    assert val == pytest.approx(mird17val, 0.4) #large acceptance in brem tail
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['Y90'], i=0,j=0,k=0,dx_mm=6.,dy_mm=6.,dz_mm=6.)    
    mird17val = 3.46E-1
    assert val == pytest.approx(mird17val, 0.02)
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['Y90'], i=5,j=5,k=5,dx_mm=6.,dy_mm=6.,dz_mm=6.)    
    mird17val = 1.39E-7
    assert val == pytest.approx(mird17val, 0.5) #large acceptance in brem tail
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['Tc99m'], i=0,j=0,k=0,dx_mm=6.,dy_mm=6.,dz_mm=6., Ns=2E7)    
    mird17val = 1.2E-2
    assert val == pytest.approx(mird17val, 0.07)
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['Tc99m'], i=5,j=5,k=5,dx_mm=6.,dy_mm=6.,dz_mm=6.)    
    mird17val = 2.33E-6
    assert val == pytest.approx(mird17val, 0.07)
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['I131'], i=0,j=0,k=0,dx_mm=6.,dy_mm=6.,dz_mm=6., Ns=2E7)    
    mird17val = 1.29E-1
    assert val == pytest.approx(mird17val, 0.03)
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['I131'], i=5,j=5,k=5,dx_mm=6.,dy_mm=6.,dz_mm=6.)    
    mird17val = 6.1E-6
    assert val == pytest.approx(mird17val, 0.03)
    
def test_resample_against_MIRD17_3mm_source_voxel():
    """
    MIRD 17 data taken from Bolch et al JNM 1999 tables. 
    
    'P-32' mGy/MBq/s
    (0,0,0) = 1.65
    (5,5,5) = 4.16E-7
    
    'Sr-89'
    (0,0,0) = 1.55
    (5,5,5) = 3.15E-7
    
    'Y-90'
    (0,0,0) = 1.61
    (5,5,5) = 6.31E-7
    
    'Tc-99m'
    (0,0,0) = 9.02E-2
    (5,5,5) = 9.12E-6

    'I-131'
    (0,0,0) = 9.2E-1
    (5,5,5) = 2.63E-5
    

    Returns
    -------
    None.

    """
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['P32'], i=0,j=0,k=0,dx_mm=3.,dy_mm=3.,dz_mm=3.)    
    mird17val = 1.65
    assert val == pytest.approx(mird17val, 0.02)
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['P32'], i=5,j=5,k=5,dx_mm=3.,dy_mm=3.,dz_mm=3.)    
    mird17val = 4.16E-7
    assert val == pytest.approx(mird17val, 0.55) #large acceptance in brems tail!
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['Sr89'], i=0,j=0,k=0,dx_mm=3.,dy_mm=3.,dz_mm=3.)    
    mird17val = 1.55
    assert val == pytest.approx(mird17val, 0.02)
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['Sr89'], i=5,j=5,k=5,dx_mm=3.,dy_mm=3.,dz_mm=3.)    
    mird17val = 3.15E-7
    assert val == pytest.approx(mird17val, 0.55) #large acceptance in brem tail
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['Y90'], i=0,j=0,k=0,dx_mm=3.,dy_mm=3.,dz_mm=3.)    
    mird17val = 1.61
    assert val == pytest.approx(mird17val, 0.03)
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['Y90'], i=5,j=5,k=5,dx_mm=3.,dy_mm=3.,dz_mm=3.)    
    mird17val = 6.31E-7
    assert val == pytest.approx(mird17val, 0.5) #large acceptance in brem tail
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['Tc99m'], i=0,j=0,k=0,dx_mm=3.,dy_mm=3.,dz_mm=3., Ns=2E7)    
    mird17val = 9.02E-2
    assert val == pytest.approx(mird17val, 0.07)
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['Tc99m'], i=5,j=5,k=5,dx_mm=3.,dy_mm=3.,dz_mm=3.)    
    mird17val = 9.12E-6
    assert val == pytest.approx(mird17val, 0.14)
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['I131'], i=0,j=0,k=0,dx_mm=3.,dy_mm=3.,dz_mm=3., Ns=2E7)    
    mird17val = 9.2E-1
    assert val == pytest.approx(mird17val, 0.03)
    
    val = approximate_voxel_kernel_from_radial_point_dose_file(filemap['I131'], i=5,j=5,k=5,dx_mm=3.,dy_mm=3.,dz_mm=3.)    
    mird17val = 2.63E-5
    assert val == pytest.approx(mird17val, 0.09)
    
def test_resample_against_DPM_Y90():
    """
        includes a the source voxel comparison for 3mm isotropic from Dose Planning Method.
        Also includes a source voxel comparison for an anisotropic source voxel 4mmx4mmx3mm. 
        DPMs run by Ben Van in spring of 2022.  
        
        Reference:
            Wilderman SJ, Dewaraja YK. Method for Fast CT/SPECT-Based 3D Monte Carlo Absorbed Dose Computations in Internal Emitter Therapy. IEEE Trans Nucl Sci. 2007 Feb 17;54(1):146-151. doi: 10.1109/TNS.2006.889164. PMID: 20305792; PMCID: PMC2841294.
        """
    val = approximate_voxel_kernel_from_radial_point_dose_file(
        filemap['Y90'], i=0, j=0, k=0, dx_mm=3., dy_mm=3., dz_mm=3.)
    expectedval = 1.581
    assert val == pytest.approx(expectedval, 0.02)

    #look at anisotropic voxel
    val = approximate_voxel_kernel_from_radial_point_dose_file(
        filemap['Y90'], i=0, j=0, k=0, dx_mm=4., dy_mm=4., dz_mm=3.)
    expectedval = 1.013
    assert val == pytest.approx(expectedval, 0.04)
