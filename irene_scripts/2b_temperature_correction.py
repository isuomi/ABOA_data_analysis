import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset

#-----------------------------------------------------------------------------------------------
#
# 19.10.2017 Irene Suomi
#
# THIS SCRIPT CORRECTS THE SONIC TEMPERATURE FOR THE CROSS-WIND EFFECT (Liu et al., 2001)
#
# This is only needed for USA-1 sonic anemometer from Aboa campaigns 2010-11 and 2014-15.
# Do not apply this to CSAT sonic anemometers, for them the correction is done internally.
#
# The original uncorrected temperature is copied and saved as parameter
# T_USA, and if T_USA already exists (in case the script is rerun), the
# correction is applied to it.
#
# The final corrected temperature is saved as parameter T.
#
#-----------------------------------------------------------------------------------------------


###########################################################################################
# data path
# path to original data
datdir = '/media/suomi/My Passport/ABOA_2010-2011/2010-Antarct/00DATA/AWS5/Flux/nc/'

# Read data file names
import glob
fnames = np.sort(np.array(glob.glob(datdir+'*.nc')))

# loop over data files
for fname in fnames:
  nc = Dataset(fname,'r+',format='NETCDF4')
  print "**********************************"
  print fname[-16:-3]
  #
  print nc.variables.keys()
  # read horizontal wind components
  x = nc.variables['X'][:]
  y = nc.variables['Y'][:]
  U2 = x**2.0 + y**2.0
  #
  # Create a new variable 
  if 'T_USA' in nc.variables.keys():
    T = nc.variables['T_USA']
  else:
    T_USA = nc.createVariable('T_USA', 'f4', ('dim'))
    T = nc.variables['T']
    T_USA[:] = T[:].copy()
  #
  # The correction function for the USA-1 sonic anemometer:
  T[:] = T[:].copy() + 3.0/(4.0*403.0)*U2
  #
  # If specific humidity is available, the crosswind corrected sonic temperature (above),
  # which is close to virtual temperature, can be corrected for humidity effects to get
  # the true air temperature:
  # T[:] = T[:].copy()/(1.0+0.51*q)
  # where q is the specific humidity.
  #
  nc.close()
