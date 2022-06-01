# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Justin Mikell, mikell@wustlDOTedu

Contributing Authors:
    Justin Mikell, PhD (Washington University St. Louis & University of Michigan)
    Benjamin Van, MS   (University of Michigan)
    
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate




def read_graves_csv(file):
   """ reads in csv and formats the dataframe to convenient headers. 
    INPUTS: a string representing a file location
    OUTPUT: a pandas dataframe
    
   """
   df = pd.read_csv(file, header=1, names = ['r_cm','E_MeV', 'E_uncert', 'mass_g', 'dose_MeV_per_g' ])
   return(df)
   
def approximate_voxel_kernel_from_radial_point_dose_file(infilelist, i=0,j=0,k=0,dx_mm=3.,dy_mm=3.,dz_mm=3.,Ns=5E6, source='UNIFORM', combine_infiles=True, verbose=0):
    """
    This function integrates going from a 1D radial dose point kernel to a dose voxel kernel with user adjustable voxel sizes. 
    
    
    JKM 5/4/22
    
    Good test cases to convince you it works: 
        replicate MIRD17 DVK  3mm isotropic 
        replicate MIRD17 DVK  6 mm isotropic. 
        comparison with DPM for anisotropic voxel sizes (which prompted this calculation)
        
    
    INPUTS:
        infilelist: a list of string pointing to the location of a csv file from Graves et al (see references below)
        i: integer, index of voxel to estimate in the x direction (0 corresponds to source)
        j: integer, index of voxel to estimate in the y direction (0 corresponds to source)
        k: integer, index of voxel to estimate in the z direction (0 corresponds to source)
        dx_mm: size of voxel in x direction in mm 
        dy_mm: size of voxel in y direction in mm
        dz_mm: size of voxel in z direction in mm
        Ns: integer, number of realizations to use in the integration. 
        source: 'UNIFORM' or 'POINT'. 
            'UNIFORM' assumes a uniform distribution of activity in the source voxel.     
            'POINT' assumes a point source of activity at center of source voxel. 
        verbose: a non-zero value will print information to stdout
        
    OUTPUTS:
        the dose in the specified voxel in mGy/MBq-s 
    
    
    WARNINGS:
        Use at your own risk. 
        There will be a lower limit of regarding voxel size that will become inaccurate. 
    
    
    EXAMPLE 1 multiple files (total dose to source voxel):
        
        file_electron = r'C:/Users/mikell/OneDrive - Washington University in St. Louis/research/MICH/20190213_Graves_DosePointKernels_v1.0/131I_8.0252D_electron.io_processed.csv'
        file_gamma = r'C:/Users/mikell/OneDrive - Washington University in St. Louis/research/MICH/20190213_Graves_DosePointKernels_v1.0/131I_8.0252D_gamma.io_processed.csv'
        file_beta = r'C:/Users/mikell/OneDrive - Washington University in St. Louis/research/MICH/20190213_Graves_DosePointKernels_v1.0/131I_8.02070D_beta.io_processed.csv'
        filelist = [file_electron, file_gamma, file_beta]
        d_000 = approximate_voxel_kernel_from_radial_point_dose_file(filelist, 0,0,0,3.0,3.0,3.0,Ns=5E6,source='UNIFORM',combine_infiles=True,verbose=0)
        
    References:
        Graves et al Dose point Kernels for 2,174 radionuclides, Medical Physics 2019 Nov; 46(11):5284-5293  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7685392/
        Their radial dose data is available at https://zenodo.org/record/2564036 as of 5/23/22.
        
    """
    

    #read in graves finely sampled kernel
    #file =  r'C:/Users/mikell/OneDrive - Washington University in St. Louis/research/MICH/EANM2022_Van/90Y_64.10H_beta.io_processed.csv'
    n_files = len(infilelist)
    if(n_files > 1):
        
        if(combine_infiles == True):
            
            graves_csv_lines = 2898
            dose_arr = np.zeros(graves_csv_lines)
            for ii in infilelist:
                dfTemp = read_graves_csv(ii)
                dose_arr += dfTemp.dose_MeV_per_g.values
            dfTemp.dose_MeV_per_g = dose_arr
            df = dfTemp
        else:
            dose_arr = np.zeros(n_files)
            for ii in np.arange(n_files):
                dose_arr[ii] = approximate_voxel_kernel_from_radial_point_dose_file([infilelist[ii]], i=i,j=j,k=k,dx_mm=dx_mm,dy_mm=dy_mm,dz_mm=dz_mm,Ns=Ns, source=source, combine_infiles=combine_infiles, verbose=verbose)
                
            return(dose_arr)
    else:
        file = infilelist[0]        
        #file =  r'C:/Users/mikell/OneDrive - Washington University in St. Louis/research/MICH/EANM2022_Van/90Y_64.10H_beta.io_processed.csv'
        df = read_graves_csv(file)
    
    
    
    dose_MeV_per_g = df.dose_MeV_per_g.values
    r_cm = df.r_cm.values
    
    
    assert np.min(r_cm) >= 0 
    
    source = str.upper(source)
    assert ('UNIFORM'==source) | ('POINT'==source)
    
    
    r_mm = 10.0*r_cm
    r_mm_center = 0.5*r_mm[0:-1] + 0.5*r_mm[1:]
    r_mm_center = np.insert(r_mm_center, 0,0)
    
    
 
    #sample source position
    #of samples:
    Ns = np.int64(Ns)
    np.random.seed()
    voxel_x_size_mm = dx_mm
    voxel_y_size_mm = dy_mm
    voxel_z_size_mm = dz_mm

    if(source == 'UNIFORM'):
        sx = np.random.uniform(-1*voxel_x_size_mm/2.0, voxel_x_size_mm/2.0, Ns)
        sy = np.random.uniform(-1*voxel_y_size_mm/2.0, voxel_y_size_mm/2.0, Ns)
        sz = np.random.uniform(-1*voxel_z_size_mm/2.0, voxel_z_size_mm/2.0, Ns)
    else: #point
        sx = np.zeros(Ns)
        sy = np.zeros(Ns)
        sz = np.zeros(Ns)
    

    min_vox_x_mm = -1*voxel_x_size_mm/2.0 + i*voxel_x_size_mm
    min_vox_y_mm = -1*voxel_y_size_mm/2.0 + j*voxel_y_size_mm
    min_vox_z_mm = -1*voxel_z_size_mm/2.0 + k*voxel_z_size_mm
    max_vox_x_mm = min_vox_x_mm + voxel_x_size_mm
    max_vox_y_mm = min_vox_y_mm + voxel_y_size_mm
    max_vox_z_mm = min_vox_z_mm + voxel_z_size_mm
    px = np.random.uniform(min_vox_x_mm, max_vox_x_mm, Ns)
    py = np.random.uniform(min_vox_y_mm, max_vox_y_mm, Ns)
    pz = np.random.uniform(min_vox_z_mm, max_vox_z_mm, Ns)
    
    sxyz = np.vstack((sx,sy,sz)).transpose()
    pxyz = np.vstack((px,py,pz)).transpose()
    
    
    q = np.power(pxyz - sxyz,2)
    q = np.sqrt(q[:,0]+q[:,1]+q[:,2])

    f = interpolate.interp1d(r_mm_center, dose_MeV_per_g, kind='linear', fill_value=(r_mm_center[0], r_mm_center[-1]))
    
    d = f(q)*0.1602

    if(verbose > 0):
        
        print("dose at i,j,k={0},{1},{2} for dx={3},dy={4},dz={5} mm voxels = {6} mGy/MBq-s".format(i,j,k,voxel_x_size_mm,voxel_y_size_mm,voxel_z_size_mm, np.mean(d)))
        print("min, max, avg, std dose (mGy/MBq-s), std/mean: {0},{1},{2},{3},{4}".format(np.min(d), np.max(d), np.mean(d), np.std(d), np.std(d)/np.mean(d)))
        print("min, max, avg, std radius (mm), std/mean: {0},{1},{2},{3},{4}".format(np.min(q), np.max(q), np.mean(q), np.std(q), np.std(q)/np.mean(q)))
        print('std error of the mean:{0}'.format(np.std(d)/np.sqrt(Ns)))
    
    return(np.mean(d))










