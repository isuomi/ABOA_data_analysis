import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import glob

###########################################################################################

# data path
datdir = '/media/suomi/My Passport/ABOA_2010-2011/2010-Antarct/00DATA/AWS5/Flux/nc/10min/'

# Read data file names
fnames = np.sort(np.array(glob.glob(datdir+'AWS5_sonic_20hz_10m_*.nc')))

print "2D coordinate transformation"

# loop over data files
for fname in [fnames]:
  #
  nc = Dataset(fname,'r+',format='NETCDF4')
  #
  keys = nc.groups.keys() # loop over heights
  for k in keys:
    #
    # read variables
    x = nc.variables['X'][:]
    y = nc.variables['Y'][:]
    z = nc.variables['Z'][:]
    #
    # COORDINATE TRANSFORMATION: 2D method
    #
    # yaw-correction: <v> = 0
    Xmn = x.mean()
    Ymn = y.mean()
    Zmn = z.mean()
    # components of the untilt tensor:
    A11 = Xmn/np.sqrt(Xmn**2.0+Ymn**2.0)
    A21 = Ymn/np.sqrt(Xmn**2.0+Ymn**2.0)
    A12 = -Ymn/np.sqrt(Xmn**2.0+Ymn**2.0)
    A22 = Xmn/np.sqrt(Xmn**2.0+Ymn**2.0)
    Ayaw = np.array([[A11,A12,0.0],[A21,A22,0.0],[0.0,0.0,1.0]])
    xx,yy,zz = np.dot( Ayaw.T, np.array([x, y, z]))
    #
    # pitch-correction: <w> = 0
    Xmn = np.mean(xx)
    Ymn = np.mean(yy)
    Zmn = np.mean(zz)
    # components of the untilt tensor:
    A11 = Xmn/np.sqrt(Xmn**2.0+Zmn**2.0)
    A31 = Zmn/np.sqrt(Xmn**2.0+Zmn**2.0)
    A13 = -Zmn/np.sqrt(Xmn**2.0+Zmn**2.0)
    A33 = Xmn/np.sqrt(Xmn**2.0+Zmn**2.0)
    Apitch = np.array([[A11,0.0,A13],[0.0,1.0,0.0],[A31,0.0,A33]])
    u,v,w = np.dot( Apitch.T, np.array([xx, yy, zz]))
    #
    print fname[-26:-3], k, u.mean(), v.mean(), w.mean()
    #
    # save results to nc-file:
    vnams = nc.variables.keys()
    if 'u' in vnams:
      su = nc.variables['u']
    else:
      su = nc.createVariable('u', 'f4', ('dim'))
    if 'v' in vnams:
      sv = nc.variables['v']
    else:
      sv = nc.createVariable('v', 'f4', ('dim'))
    if 'w' in vnams:
      sw = nc.variables['w']
    else:
      sw = nc.createVariable('w', 'f4', ('dim'))
    su[:] = u
    sv[:] = v
    sw[:] = w
    #
  nc.close()
