import numpy as np
from matplotlib import pyplot as plt
from netCDF4 import Dataset
import datetime
import os
import glob

###########################################################################################
# functions

#------------------------------------------------------------------------------------
def movingaverage(interval, window_size):
    window= np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def cal_gust(U,tg):
    u_move = movingaverage(U, tg*20.0)[int(tg*10):-int(tg*10)]
    Umax = np.ma.max(u_move)
    Umin = np.ma.min(u_move)
    return Umax, Umin

def cal_gust_components(u,v,w,tg):
    #
    # horizontal wind speed array
    U = np.sqrt(u**2.0+v**2.0)
    #
    # moving average of the horizontal wind speed, with a window length = gust length tg
    U_move = movingaverage(U, tg*20.0)[int(tg*10):-int(tg*10)]
    #
    # maximum and minimum gusts
    Umax = np.ma.max(U_move)
    #
    # minimum gusts
    Umin = np.ma.min(U_move)
    #
    # the location of the first occurrence of the gusts
    loc_max = np.where(U_move==Umax)[0][0]
    loc_min = np.where(U_move==Umin)[0][0]
    #
    # moving averages of the wind components:
    u_move = movingaverage(u, tg*20.0)[int(tg*10):-int(tg*10)]
    v_move = movingaverage(v, tg*20.0)[int(tg*10):-int(tg*10)]
    w_move = movingaverage(w, tg*20.0)[int(tg*10):-int(tg*10)]
    #
    # return the maximum and the minimum gusts + the wind components 
    return Umax, u_move[loc_max], v_move[loc_max], w_move[loc_max], Umin, u_move[loc_min], v_move[loc_min], w_move[loc_min]

def WindDirection(u,v):
    wdir=np.array(u)*np.nan
    # right-handed coordinate system:
    # u is positive to east
    # v is positive to north
    if np.size(u)>1:
        i=v!=0.0
        wdir[i]=np.arctan(u[i]/v[i])/np.pi*180.0
        i=( ( (u<=0.0) & (v>0.0) ) | ( (u>0.0) & (v>0.0) ) )
        wdir[i]+=180.0
        i=( (u>0.0) & (v<0.0) )
        wdir[i]+=360.0
        i=( (v==0.0) & (u<0.0) )
        wdir[i]=90.0
        i=( (v==0.0) & (u>0.0) )
        wdir[i]=270.0
    else:
        if v!=0.0:
            wdir=np.arctan(u/v)/np.pi*180.0
        if ( ( (u<=0.0) & (v>0.0) ) | ( (u>0.0) & (v>0.0) ) ):
            wdir+=180.0
        if ( (u>0.0) & (v<0.0) ):
            wdir+=360.0
        if ( (v==0.0) & (u<0.0) ):
            wdir=90.0
        if ( (v==0.0) & (u>0.0) ):
            wdir=270.0
    return wdir


###########################################################################################
#
# Read data from here:
datdir = '/media/suomi/My Passport/ABOA_2010-2011/2010-Antarct/00DATA/AWS5/Flux/nc/10min/'

# Save results here:
resdir = '/media/suomi/My Passport/ABOA_2010-2011/2010-Antarct/00DATA/AWS5/Flux/mean/'
#
# Create a time array with all 10 min periods from 
t = np.arange('2010-12-16T00:00Z', '2011-01-23T00:00Z',np.timedelta64(10,'m'), dtype='datetime64')
print "Time interval is from",t[0],"to",t[-1],", Array size is",t.shape
#
# Create output file for the results:
if os.path.isfile(resdir+'Mean_data_2010-2011.nc'):
    nc = Dataset(resdir+'Mean_data_2010-2011.nc','r+',format='NETCDF4')
    #
    #--------------------------------------------
    # Open variables
    #
    # Mean from sonic
    sTime = nc.variables['time']
    sU = nc.variables['U']
    sX = nc.variables['X']
    sY = nc.variables['Y']
    sDir = nc.variables['dir']
    sStd_U = nc.variables['std_U']
    sStd_u = nc.variables['std_u']
    sStd_v = nc.variables['std_v']
    sStd_w = nc.variables['std_w']
    sTKE = nc.variables['TKE']
    sT_S = nc.variables['T_sonic']
##    sTheta_S = nc.variables['theta_sonic']    # if pressure data available
##    sPres = nc.variables['p']                 # if pressure data available
    #
    #-----------------------------------
    # Fluxes
    sUW = nc.variables['uw']
    sVW = nc.variables['vw']
    sUstar = nc.variables['ustar']
    sWT = nc.variables['wT']
##    sWTheta = nc.variables['wtheta']  # if potential temperature available
    s_invL = nc.variables['invL']
    #
    #-----------------------------------
    # Gusts
    s_Umax = nc.variables['Umax']
    s_u_max = nc.variables['u_max']
    s_v_max = nc.variables['v_max']
    s_w_max = nc.variables['w_max']
    #
    s_Umin = nc.variables['Umin']
    s_u_min = nc.variables['u_min']
    s_v_min = nc.variables['v_min']
    s_w_min = nc.variables['w_min']
    #
    #
else:
    # create file
    nc = Dataset(resdir+'Mean_data_2010-2011.nc','w',format='NETCDF4')
    #
    # Create dimensions
    nc.createDimension('dim_time',len(t))
    nc.createDimension('dim_hgt',2) # save results for both sonic heights here
##    nc.createDimension('dim_gust',N_gusts) # gusts can be calculated with various lengths, now we use only 3s
    #
    #--------------------------------------------
    # Create variables
    #
    # Mean from sonic
    Time = nc.createVariable('time','f4',('dim_time'))
    Time[:] = t
    sU = nc.createVariable('U','f4',('dim_time','dim_hgt'))
    sX = nc.createVariable('X','f4',('dim_time','dim_hgt'))
    sY = nc.createVariable('Y','f4',('dim_time','dim_hgt'))
    sDir = nc.createVariable('dir','f4',('dim_time','dim_hgt'))
    sStd_U = nc.createVariable('std_U','f4',('dim_time','dim_hgt'))
    sStd_u = nc.createVariable('std_u','f4',('dim_time','dim_hgt'))
    sStd_v = nc.createVariable('std_v','f4',('dim_time','dim_hgt'))
    sStd_w = nc.createVariable('std_w','f4',('dim_time','dim_hgt'))
    sTKE = nc.createVariable('TKE','f4',('dim_time','dim_hgt'))
    sT_S = nc.createVariable('T_sonic','f4',('dim_time','dim_hgt'))
##    sTheta_S = nc.createVariable('theta_sonic','f4',('dim_time','dim_hgt'))
##    sPres = nc.createVariable('p','f4',('dim_time','dim_hgt'))
    #
    #-----------------------------------
    # Fluxes
    sUW = nc.createVariable('uw','f4',('dim_time','dim_hgt'))
    sVW = nc.createVariable('vw','f4',('dim_time','dim_hgt'))
    sUstar = nc.createVariable('ustar','f4',('dim_time','dim_hgt'))
    sWT = nc.createVariable('wT','f4',('dim_time','dim_hgt'))
##    sWTheta = nc.createVariable('wtheta','f4',('dim_time','dim_hgt')) # if potential temperature available
    s_invL = nc.createVariable('invL','f4',('dim_time','dim_hgt'))
    #
    #-----------------------------------
    # Gusts
    s_Umax = nc.createVariable('Umax','f4',('dim_time','dim_hgt'))#,'dim_gust'))
    s_u_max = nc.createVariable('u_max','f4',('dim_time','dim_hgt'))#,'dim_gust'))
    s_v_max = nc.createVariable('v_max','f4',('dim_time','dim_hgt'))#,'dim_gust'))
    s_w_max = nc.createVariable('w_max','f4',('dim_time','dim_hgt'))#,'dim_gust'))
    #
    s_Umin = nc.createVariable('Umin','f4',('dim_time','dim_hgt'))#,'dim_gust'))
    s_u_min = nc.createVariable('u_min','f4',('dim_time','dim_hgt'))#,'dim_gust'))
    s_v_min = nc.createVariable('v_min','f4',('dim_time','dim_hgt'))#,'dim_gust'))
    s_w_min = nc.createVariable('w_min','f4',('dim_time','dim_hgt'))#,'dim_gust'))
    #

#-----------------------------
#
# Some constants/parameters needed later
#
kappa = 0.4 # von Karman constant
g = 9.81 # gravitational acceleration
Rd = 287.0 # gas constant for dry air
cp = 1004.0 # specific heat of dry air
##rho = (p2*100.0)/(Rd*(T2+273.15)) # air density; 2010-11 campaign: pressure is not available
#
# read file names
fnames = np.sort(np.array(glob.glob(datdir+'AWS5_sonic_20hz_10m_*.nc')))
lev = 1 # height index for 10 m level
##lev = 0 # height index for 2 m level
#
#-----------------------------
# Initialize parameters:
#
# gust lengths
tg = 3.0 #np.arange(1.0,31.1,1.0)
#
print "--------------------------------------------------"

# Read all nc-files
fname_inds = range(len(fnames))
for i in fname_inds:#[:1000:]:
    #
    # read time from the filename
    tim = fnames[i][-15:-3]
    print tim
    YY = tim[:4]
    MM = tim[4:6]
    DD = tim[6:8]
    HH = tim[8:10]
    MI = tim[10:]
    t_dat = np.datetime64(YY+'-'+MM+'-'+DD+'T'+HH+':'+MI+'Z')
    #
    # find the location of the data point on the time axis
    msk_t = t==t_dat
    if len(msk_t[msk_t])==1:
        #
        # Open the nc-file:
        nc_in = Dataset(fnames[i],'r',format='NETCDF4')
##        status = nc_in.variables['status'][:] # use this variable "status" to check the data quality 
##        msk_status = status > 0 # if status flag is equal to 0, the data quality is good
        # if (len(msk_status[-msk_status])>12000*0.999): # this can be a criteria for good data
        #
        #-------------------
        # calculate mean variables
        u = nc_in.variables['u'][:] # read in u-velocity
        v = nc_in.variables['v'][:] # read in v-velocity
        w = nc_in.variables['w'][:] # read in w-velocity
        U = np.sqrt(u**2.0+v**2.0) # horizontal wind speed
        sX[msk_t,lev] = nc_in.variables['X'][:].mean() # mean of X-velocity
        sY[msk_t,lev] = nc_in.variables['Y'][:].mean() # mean of Y-velocity
        # WIND DIRECTION: CHECK WHERE X AND Y ARE POINTING TO GET THE TRUE WIND DIRECTION!!
        sDir[msk_t,lev] = WindDirection(sY[msk_t,lev],sX[msk_t,lev])
        sU[msk_t,lev] = u.mean()
        T = nc_in.variables['T'][:]+273.15
        sT_S[msk_t,lev] = T.mean()
        #
        #-------------------
        # standard deviations
        sStd_U[msk_t,lev] = np.std(U)
        sStd_u[msk_t,lev] = np.std(u)
        sStd_v[msk_t,lev] = np.std(v)
        sStd_w[msk_t,lev] = np.std(w)
        #
        #-------------------
        # Fluxes and the Obukhov length
        #
        # friction velocity
        sUW[msk_t,lev] = ((u-u.mean())*w).mean()
        sVW[msk_t,lev] = (v*w).mean()
        sUstar[msk_t,lev] = (sUW[msk_t,lev]**2.0+sVW[msk_t,lev]**2.0)**(1./4.)
        #
##        # pressure and (virtual) potential temperature at mast levels
##        sPres[msk_t,lev] = p2[msk_t] - rho[msk_t] * g * (z_levs[lev] - 2.0)
##        theta = (T+273.15)*(1000.0/sPres[msk_t,lev])**(Rd/cp)
##        sTheta_S[msk_t,lev] = theta.mean()
        #
        # heat flux (virtual)
##        sWTheta[msk_t,lev] = (w*(theta-theta.mean())).mean()
        sWT[msk_t,lev] = (w*(T-T.mean())).mean()
        #
        # Obukhov length (inverse of it)
        s_invL[msk_t,lev] = ( - kappa * g / sT[msk_t,lev] * sWT[msk_t,lev] / (sUstar[msk_t,lev]**3.0) )
##        s_invL[msk_t,lev] = ( - kappa * g / sTheta_S[msk_t,lev] * sWTheta[msk_t,lev] / (sUstar[msk_t,lev]**3.0) )
        #
        #-------------------
        # gusts
        s_Umax[msk_t,lev], s_u_max[msk_t,lev], s_v_max[msk_t,lev], \
                           s_w_max[msk_t,lev], s_Umin[msk_t,lev], \
                           s_u_min[msk_t,lev], s_v_min[msk_t,lev], \
                           s_w_min[msk_t,lev]= cal_gust_components(u,v,w,tg)
    nc_in.close()
#
#


nc.close()
#
print "**********************************"
#
