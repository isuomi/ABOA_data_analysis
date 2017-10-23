import matplotlib.pylab as plt
import numpy as np
import os.path
from netCDF4 import Dataset
import glob

# path to original data
datdir_10 = '/media/suomi/My Passport/ABOA_2010-2011/2010-Antarct/00DATA/AWS5/Flux/nc/10min/'
# path to netcdf data for samples:
outdir_30 = '/media/suomi/My Passport/ABOA_2010-2011/2010-Antarct/00DATA/AWS5/Flux/nc/30min/'

# ascii file names
fnames = np.sort(np.array(glob.glob(datdir_10+'AWS5_sonic_20hz_10m_*.nc')))

time = np.array([],dtype=int)
HH = np.array([],dtype=int)
MI = np.array([],dtype=int)

# Loop over ascii files:
for fname in fnames:#[fnames[0]]:
    #
    # read hours and minutes from file names:
    HH = np.append(HH,int(fname[-7:-5]))
    MI = np.append(MI,int(fname[-5:-3]))
#
# pick up all hours available
hrs = np.unique(HH) 
print hrs

# loop over different hours:
for h in hrs:
    print h
    # this loop is to choose between the two 30 min periods within the hour
    for i in range(2):
        if i == 0:
            msk = ((HH==h)&((MI==0)|(MI==10)|(MI==20)))
        else:
            msk = ((HH==h)&((MI==30)|(MI==40)|(MI==50)))
        #
        # chack that all 10 min data are available to continue
        if len(msk[msk])==3:
            files = fnames[msk]
            tle = files[0][-15:-3] # time for output file name
            #
            # collect data from the three 10 min data files
            t = np.array([],dtype=int)
            jno = np.array([],dtype=int)
            X = np.array([],dtype=int)
            Y = np.array([],dtype=int)
            Z = np.array([],dtype=int)
            T = np.array([],dtype=int)
            #
            n=0
            for f in files:
                nc = Dataset(f,'r')
                t = np.append(t,nc.variables['time'][:])
                jno = np.append(jno,n*12000+nc.variables['jno'][:])
                X = np.append(X,nc.variables['X'][:])
                Y = np.append(Y,nc.variables['Y'][:])
                Z = np.append(Z,nc.variables['Z'][:])
                T = np.append(T,nc.variables['T'][:])
                n+=1
            #
            # save data
            # create/open nc-file
            if os.path.isfile(outdir_30+'AWS5_sonic_20hz_10m_'+tle+'.nc'):
                nc_out = Dataset(outdir_30+'AWS5_sonic_20hz_10m_'+tle+'.nc','r+',format='NETCDF4')
            else:
                nc_out = Dataset(outdir_30+'AWS5_sonic_20hz_10m_'+tle+'.nc','w',format='NETCDF4')
            #
            # create dimension
            dimkeys = nc_out.dimensions.keys()
            if ('dim') not in dimkeys:
                nc_out.createDimension('dim',len(t))
            #
            # create variables
            varkeys = nc_out.variables.keys()
            if ('time') in varkeys:
                time_save = nc_out.variables['time']
            else:
                time_save = nc_out.createVariable('time', 'i8', ('dim'))
            if ('jno') in varkeys:
                jno_save = nc_out.variables['jno']
            else:
                jno_save = nc_out.createVariable('jno', 'i8', ('dim'))
            if ('X') in varkeys:
                x_save = nc_out.variables['X']
            else:
                x_save = nc_out.createVariable('X', 'f4', ('dim'))
            if ('Y') in varkeys:
                y_save = nc_out.variables['Y']
            else:
                y_save = nc_out.createVariable('Y', 'f4', ('dim'))
            if ('Z') in varkeys:
                z_save = nc_out.variables['Z']
            else:
                z_save = nc_out.createVariable('Z', 'f4', ('dim'))
            if ('T') in varkeys:
                T_save = nc_out.variables['T']
            else:
                T_save = nc_out.createVariable('T', 'f4', ('dim'))
            #
            # save data into the nc-file:
            time_save[:] = t
            x_save[:] = X
            y_save[:] = Y
            z_save[:] = Z
            T_save[:] = T
            nc_out.close()
        
