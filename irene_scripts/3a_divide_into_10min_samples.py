import matplotlib.pylab as plt
import numpy as np
import os.path
from netCDF4 import Dataset
import glob

# path to original data
datdir = '/media/suomi/My Passport/ABOA_2010-2011/2010-Antarct/00DATA/AWS5/Flux/nc/'
# path to netcdf data for samples:
outdir_10 = '/media/suomi/My Passport/ABOA_2010-2011/2010-Antarct/00DATA/AWS5/Flux/nc/10min/'

# read file names
fnames = np.sort(np.array(glob.glob(datdir+'AWS5_sonic_20hz_10m_*.nc')))

# Loop over ascii files:
for fname in [fnames[0]]:
    #
    print fname
    #
    # Read data
    nc = Dataset(fname,'r')
    print nc.variables.keys()
    time = nc.variables['time'][:]
    x = nc.variables['X'][:]
    y = nc.variables['Y'][:]
    z = nc.variables['Z'][:]
    T = nc.variables['T'][:]
    nc.close()
    #
    # Split data into 10 min sequencies:
    u10min = np.unique(np.floor(time[:]/1.0e+3-np.floor(time[:]/1.0e+6)*1.0e+3))
    #
    for i in u10min:
        #
        # Find all data points belonging to each 10 min sequence:
        msk = np.floor(time[:]/1.0e+3-np.floor(time[:]/1.0e+6)*1.0e+3)==i
        #
        # check if the number of data points is correct (10 min of 20 Hz data):
        if len(msk[msk])==10*60*20:
            t10 = time[msk]
            x10 = x[msk]
            y10 = y[msk]
            z10 = z[msk]
            T10 = T[msk]
            jno = np.arange(len(t10),dtype=int)+1 # order number of the sample
            #
            tlabel = '20'+np.floor(t10[0]/1.0e+6).astype(int).astype(str)
            if i<10:
                tlabel = tlabel+'00'+str(int(i))+'0'
            elif i<100:
                tlabel = tlabel+'0'+str(int(i))+'0'
            else:
                tlabel = tlabel+str(int(i))+'0'
            #
            print i,jno[-1],tlabel
            #
            # save data
            # create/open nc-file
            if os.path.isfile(outdir_10+'AWS5_sonic_20hz_10m_'+tlabel+'.nc'):
                nc = Dataset(outdir_10+'AWS5_sonic_20hz_10m_'+tlabel+'.nc','r+',format='NETCDF4')
            else:
                nc = Dataset(outdir_10+'AWS5_sonic_20hz_10m_'+tlabel+'.nc','w',format='NETCDF4')
            #
            # create dimension
            dimkeys = nc.dimensions.keys()
            if ('dim') not in dimkeys:
                nc.createDimension('dim',len(t10))
            #
            # create variables
            varkeys = nc.variables.keys()
            if ('time') in varkeys:
                time_save = nc.variables['time']
            else:
                time_save = nc.createVariable('time', 'i8', ('dim'))
            if ('jno') in varkeys:
                jno_save = nc.variables['jno']
            else:
                jno_save = nc.createVariable('jno', 'i8', ('dim'))
            if ('X') in varkeys:
                x_save = nc.variables['X']
            else:
                x_save = nc.createVariable('X', 'f4', ('dim'))
            if ('Y') in varkeys:
                y_save = nc.variables['Y']
            else:
                y_save = nc.createVariable('Y', 'f4', ('dim'))
            if ('Z') in varkeys:
                z_save = nc.variables['Z']
            else:
                z_save = nc.createVariable('Z', 'f4', ('dim'))
            if ('T') in varkeys:
                T_save = nc.variables['T']
            else:
                T_save = nc.createVariable('T', 'f4', ('dim'))
            #
            # save data into the nc-file:
            time_save[:] = t10
            jno_save[:] = jno
            x_save[:] = x10
            y_save[:] = y10
            z_save[:] = z10
            T_save[:] = T10
            nc.close()
