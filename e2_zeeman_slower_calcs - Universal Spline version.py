import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline
from B_profile_generator import sample

"""
Uses the data that Kayleigh took for the E2 Zeeman Slower to optimize the
performance of the Zeeman slower and verify that we have the right profile.
Note that for simplicity, we will always have the field be "decreasing" in
value, if not in magnitude (eg, dBdz<0 in slowing region) - so that sigma-
light will need to be used. Consult the OneNote to see what kind of profiles
are being considered.
"""

### Section for global constants for our problem: note that distances will be
### measured in m, magnetic fields will be measured in Gauss, velocities in m/s,
### wavelengths in um, detunings in MHz (these two mean delta*lamda -> m/s),

# broad cooling transition parameters
ADIA_CONST = 0.6525 #(m/s) per Gauss
A_MAX = 5.52e5 #(m/s^2) of the maximum theoretical acceleration (yes really that big)
LAMBDA = 0.4983121 #(um)
DETUNE_TO_B = 0.754 #(Gauss/MHz) for converting a detuning to a equiv B field
A_CONST = 2.75e5 #(m/s^2)

# specific slower parameters
ETA = 0.6 #Fudge factor for max allowable acceleration to allow for noise/ inperfections
SAT_FACTOR = ETA/(1-ETA)
ZS_CAP_VEL = 600.0 #(m/s)
MOT_CAP_VEL = 100.0 #(m/s)
Z_AXIS = np.linspace(0,0.770,10000) #(m)
ZS_START = 0.050 #(m)
ZS_END = 0.670 #(m)
MOT_START = 0.720 #(m)
ZS_START_IND = np.abs(Z_AXIS-ZS_START).argmin()
ZS_END_IND = np.abs(Z_AXIS-ZS_END).argmin()
MOT_START_IND = np.abs(Z_AXIS-MOT_START).argmin()

FIXFLAG=True
if FIXFLAG:
    SLOWER_RESISTANCES = {'R_red': 0.0087, 'R_green': 0.0494, 'R_yellow': 0.0887, 'R_pruple': 0.0753, 'R_blue': 0.0635005}
else:
    SLOWER_RESISTANCES = {'R_red': 0.0087, 'R_green': 0.0494, 'R_yellow': 0.0887, 'R_pruple': 0.0753}

# useful lists to try to optimize over
DETUNINGS = [0,-200,-400,-600,-800,-1000,-1200,-1400,-1600]

# useful commands to run
# B_field_dict = find_ZS_profiles([[0,0,0,0,0,35] for i in DETUNINGS], Z_AXIS, DETUNINGS)
# (or) B_field_dict = find_ZS_profiles([[0,0,0,0,0] for i in DETUNINGS], Z_AXIS, DETUNINGS)
# ZS_plot(Z_AXIS, B_field_dict)

# useful mappings for current list indicies -> color of ZS
red_ind = 0
yellow_ind = 1
green_ind = 2
purple_ind = 3
# blue_ind = 4


# useful lists for plots:
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'fuchsia', 'indigo']

### Generate a functions for the magnetic field as a function of position and 
### a current vector [I_red, I_yellow, I_green, I_purple]

zs_calib_data = np.genfromtxt("slower_data_round_2_v2.csv", delimiter=',', skip_header=1).T
# if FIXFLAG:
#     fx_calib_data = np.genfromtxt("fixingCoil.csv")
#     fx_calib_data = fx_calib_data.round(2)
#     fx_calib_data = -fx_calib_data #since all data are negative, negate this as well
#     x = np.array(range(1,78))
#     fx_calib_data = np.vstack((x,fx_calib_data))
#     zs_calib_data = np.vstack((zs_calib_data,fx_calib_data))
    
red_calib_data = zs_calib_data[0:2,]
yellow_calib_data = np.flip(zs_calib_data[2:4,], axis=1)
green_calib_data = np.flip(zs_calib_data[4:6,], axis=1)
purple_calib_data = np.flip(zs_calib_data[6:8,], axis=1)

# Specific operation to red data because of nan and inverse, same for blue
red_calib_x_data = red_calib_data[0][0:np.argmax(red_calib_data[0])]
red_calib_B_data = red_calib_data[1][0:np.argmax(red_calib_data[1])]
red_calib_data = np.flip(np.reshape(np.concatenate([red_calib_x_data, red_calib_B_data]),(2,44)),axis=1)

red_B_prof = UnivariateSpline(red_calib_data[0]/100.0, red_calib_data[1]) #/100 to convert to m
yellow_B_prof = UnivariateSpline(yellow_calib_data[0]/100.0, yellow_calib_data[1])
green_B_prof = UnivariateSpline(green_calib_data[0]/100.0, green_calib_data[1])
purple_B_prof = UnivariateSpline(purple_calib_data[0]/100.0, purple_calib_data[1])
if FIXFLAG:
    bluex,bluey = sample()
    bluex = [x/100 for x in bluex]
    blue_B_prof = UnivariateSpline(bluex, bluey)

red_dBdz = red_B_prof.derivative()
yellow_dBdz = yellow_B_prof.derivative()
green_dBdz = green_B_prof.derivative()
purple_dBdz = purple_B_prof.derivative()
if FIXFLAG:
    blue_dBdz = blue_B_prof.derivative()

red_d2Bdz2 = red_B_prof.derivative(2)
yellow_d2Bdz2 = yellow_B_prof.derivative(2)
green_d2Bdz2 = green_B_prof.derivative(2)
purple_d2Bdz2 = purple_B_prof.derivative(2)
if FIXFLAG:
    blue_d2Bdz2 = blue_B_prof.derivative(2)

#splines for the ZS B fields given an input current
def exp_B_field(I, z):
    B = I[0]*red_B_prof(z)+I[1]*yellow_B_prof(z)+I[2]*green_B_prof(z)+I[3]*purple_B_prof(z)
    if FIXFLAG:
        B += I[4]*blue_B_prof(z)
    return B

def exp_B_field_fix(I, z, coilPos):
    B = I[0]*red_B_prof(z)+I[1]*yellow_B_prof(z)+I[2]*green_B_prof(z)+I[3]*purple_B_prof(z)
    if FIXFLAG:
        B += I[4]*blue_B_prof(z-coilPos)
    return B



def exp_dBdz(I, z):
    dBdz = I[0]*red_dBdz(z)+I[1]*yellow_dBdz(z)+I[2]*green_dBdz(z)+I[3]*purple_dBdz(z)
    if FIXFLAG:
        dBdz += I[4]*blue_dBdz(z)
    return dBdz

def exp_dBdz_fix(I, z, coilPos):
    dBdz = I[0]*red_dBdz(z)+I[1]*yellow_dBdz(z)+I[2]*green_dBdz(z)+I[3]*purple_dBdz(z)
    if FIXFLAG:
        dBdz += I[4]*blue_dBdz(z-coilPos)
    return dBdz

def exp_d2Bdz2(I, z):
    d2Bdz2 = I[0]*red_d2Bdz2(z)+I[1]*yellow_d2Bdz2(z)+I[2]*green_d2Bdz2(z)+I[3]*purple_d2Bdz2(z)
    if FIXFLAG:
        d2Bdz2 += I[4]*blue_d2Bdz2(z)
    return d2Bdz2

# estimate the time down the ZS
def ZS_time_est(vi):
    time = ZS_START/vi
    if (vi > ZS_CAP_VEL or vi < MOT_CAP_VEL):
        time += (Z_AXIS[-1]-ZS_START)/vi
    else:
        time += 2*(ZS_END-ZS_START)/(vi+MOT_CAP_VEL) + (Z_AXIS[-1]-ZS_END)/MOT_CAP_VEL
    return time

# B-field profile to fit for a given ZS detuning/ slowing profile
def ideal_B_field(z, detuning):
    B_field = np.zeros(len(z))
    goal_a = (ZS_CAP_VEL**2-MOT_CAP_VEL**2)/(2*(ZS_END-ZS_START))
    vi = ZS_CAP_VEL
    for i in range(ZS_START_IND, ZS_END_IND):
        B_field[i] = detuning*DETUNE_TO_B+vi*np.sqrt(1-2*goal_a*(z[i]-ZS_START)/(vi)**2)/ADIA_CONST
    return B_field

# attempts to find the endpoints of the slow region - by these points the slower
# will effecitvely stop, and we will only focus our constraints on the fit
# (due to adabaticity and Delta B) within this region.
def find_slower_endpoints(I, z, slower_type="inc_field"):
    list_len = len(z)
    rv = [0,len(z)]
    #print(rv)
    B = exp_B_field(I, z)
    d2Bdz2 = exp_d2Bdz2(I, z)
    if slower_type == "inc_field":
        rv[0] = ZS_START_IND+d2Bdz2[ZS_START_IND:int(list_len/6)].argmax()     #QUESTION: WHY 6
        rv[1] = int(3*list_len/4)+B[int(3*list_len/4):ZS_END_IND].argmin()
    if slower_type == "dec_field":
        rv[0] = ZS_START_IND+B[ZS_START_IND:int(list_len/6)].argmax()
        rv[1] = int(3*list_len/4)+d2Bdz2[int(3*list_len/4):ZS_END_IND].argmin()
    if slower_type == "spin_flip":
        rv[0] = ZS_START_IND+B[ZS_START_IND:int(list_len/6)].argmax()
        rv[1] = int(3*list_len/4)+B[int(3*list_len/4):ZS_END_IND].argmin()
    return (rv[0], rv[1])


# calculated deviation from adiabaticity along slower, to be plotted to show
# how robust the slowing ought to be
def adia_dev(I, z, detuning, start, end):
    dev = ADIA_CONST*exp_dBdz(I, z)[start:end]
    #print(dev)
    dev = dev*(detuning*LAMBDA-ADIA_CONST*ideal_B_field(z,detuning)[start:end])## check this line!
    return np.abs(dev)

# calculated deviation from adiabaticity along slower, to be plotted to show
# how robust the slowing ought to be
# with an increased I parameter
def adia_dev_fix1(I, z, detuning, start, end):
    #unpack I
    I = I.tolist()
    coilPos = I.pop()
    I = np.asarray(I)
    
    dev = ADIA_CONST*exp_dBdz_fix(I, z, coilPos)[start:end]
    #print(dev)
    dev = dev*(detuning*LAMBDA-ADIA_CONST*ideal_B_field(z,detuning)[start:end])## check this line!
    return np.abs(dev)

# calculated deviation from adiabaticity along slower, to be plotted to show
# how robust the slowing ought to be
# with a normal I but additional parameter at the end
def adia_dev_fix2(I, z, detuning, start, end, coilPos):
    dev = ADIA_CONST*exp_dBdz_fix(I, z, coilPos)[start:end]
    #print(dev)
    dev = dev*(detuning*LAMBDA-ADIA_CONST*ideal_B_field(z,detuning)[start:end])## check this line!
    return np.abs(dev)

# determine slwoer type (inc_field, dec_field, spin_flip) based on ideal B field
def det_slower_type(z, detuning):
    ideal_B = ideal_B_field(z, detuning)
    pos_field_bool = (ideal_B>0).any()
    neg_field_bool = (ideal_B<0).any()
    if (pos_field_bool and neg_field_bool):
        #print("spin_flip")
        return "spin_flip"
    elif (pos_field_bool and not(neg_field_bool)):
        #print("dec_field")
        return "dec_field"
    elif (not(pos_field_bool) and neg_field_bool):
        #print("inc_field")
        return "inc_field"
    else:
        print("foo")
        return None

# penalty function to minimize by least sq method
def func_to_minimize(I, z, detuning):
    
    #unpack I
    I = I.tolist()
    coilPos = I.pop()
    I = np.asarray(I)
    
    ideal_B = ideal_B_field(z, detuning)
    act_B = exp_B_field_fix(I, z, coilPos)
    slower_type = det_slower_type(z, detuning)
    
    #calc deviations from adabaticity
    adia_penalty = 0
    slow_reg_i, slow_reg_f = find_slower_endpoints(I, z, slower_type)
    #print(slow_reg_i)
    #print(slow_reg_f)
    deviation = adia_dev_fix2(I, z, detuning, slow_reg_i, slow_reg_f,coilPos)
    #print(deviation)
    max_adia_dev = deviation.max()
    if max_adia_dev > ETA*A_MAX:
        adia_penalty = max_adia_dev
    
    #calc capture velocity penalty
    cap_vel_penalty = 0
    act_cap_vel = np.abs(detuning*LAMBDA-ADIA_CONST*act_B[slow_reg_f])
    if act_cap_vel > MOT_CAP_VEL:
        cap_vel_penalty = 1000000
    
    #calc least squares in the ROIs (ignoring the other behavior)
    fit_penalty = np.sum((ideal_B[ZS_START_IND:ZS_END_IND]-act_B[ZS_START_IND:ZS_END_IND])**2)
    fit_penalty += np.sum((ideal_B[MOT_START_IND:]-act_B[MOT_START_IND:])**2)/2
    
    return fit_penalty + adia_penalty + cap_vel_penalty

# function to optimize the currents
def find_opt_currents(I0, z, detuning):
    bounds = [(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-np.inf,np.inf),(-10,10),(20,60)]
    res = minimize(func_to_minimize, I0, args=(z, detuning),bounds=bounds)
    print('optimized detuning='+str(detuning)+'MHz, B-field profile')
    return res['x']

# solves for optimal currents (and B fields) for a list of detunings/ inital currents
# (assumes the standard capture velocity as the goal). Also estimates the deviation
# from adiabaticity
def find_ZS_profiles(I0_list, z, detuning_list):
    zs_profiles = {}
    ind_list = range(len(I0_list))
    ideal_B_list = [ideal_B_field(z, detuning) for detuning in detuning_list]
    If_list = [find_opt_currents(I0_list[i], z, detuning_list[i]) for i in ind_list]
    print("test1")
    exp_B_list = [exp_B_field_fix(If[:-1],z,If[-1]) for If in If_list]
    print("test2")
    adia_dev_list = [adia_dev_fix1(If_list[i],z,detuning_list[i],ZS_START_IND,ZS_END_IND) for i in ind_list]
    print("test3")
    zs_profiles["ideal_B_list"] = ideal_B_list
    zs_profiles["If_list"] = If_list
    print("test4")
    zs_profiles["exp_B_list"] = exp_B_list
    zs_profiles["adia_dev_list"] = adia_dev_list
    print("test5")
    return zs_profiles

# find the acceleration of the beam at a given point in the ZS (takes all params
# into account)
def ZS_accel(z, v, I, detuning, eta):
    s0 = eta/(1-eta)
    B_curr = 0
    if (z <= Z_AXIS[-1] and z >= Z_AXIS[0]):
        B_curr = exp_B_field(I, z)
    delta = detuning + v/LAMBDA - B_curr/DETUNE_TO_B
    rv=(A_CONST/LAMBDA)*s0/(1+s0+(2.0*delta/10.5)**2)
    return rv

# defines the differential equaiton to solve with solve_ivp for the velocity
def diff_eq_for_prop(t, vec, I, detuning, eta):
    return (vec[1], -ZS_accel(vec[0], vec[1], I, detuning, eta))

# propagates a single velocity down the ZS
def prop_v(vi, I, detuning, eta):
    vec_i = np.array([0, vi])
    int_time = ZS_time_est(vi)
    rv = solve_ivp(diff_eq_for_prop, (0,int_time), vec_i, args=(I, detuning, eta), max_step=1e-7)
    return rv

# propagates a number of velocities down the ZS
def get_vel_set(vis, I, detuning, eta):
    rv={'detuning': detuning, 'eta': eta, 'I': eta/(1-eta)}
    for vi in vis:
        prop_res = prop_v(vi, I, detuning, eta)
        if prop_res.success == True:
            print('foo')
            rv[str(vi)] = {'t': prop_res.t, 'v': prop_res.y[1], 'z': prop_res.y[0]}
        else:
            print('bar')
            rv[str(vi)] = 'failed'
    return rv

# plots (and optionally saves) the ZS optimization for a given detuning.
def ZS_plot(z, ZS_dict):
    fig, axes = plt.subplots(2,1, sharex=True)
    for i in range(len(ZS_dict['ideal_B_list'])):
        ideal_B = ZS_dict['ideal_B_list'][i]
        exp_B = ZS_dict['exp_B_list'][i]
        adia_dev = ZS_dict['adia_dev_list'][i]
        col = COLORS[i]
        axes[0].plot(z, ideal_B, linestyle='dashed', color=col)
        axes[0].plot(z, exp_B, linestyle='solid', color=col)
        axes[1].plot(z[ZS_START_IND:ZS_END_IND], adia_dev/(A_MAX), color=col)
    axes[0].set_ylabel("B Field (G)")
    axes[1].set_ylabel('adiabat_cond.')
    axes[1].set_ylim(0,2)
    axes[1].axhline(1, lw=1, ls='--', color='k')
    axes[1].set_xlabel("Distance down slower (m)")
    return None

def ZS_Residual_plot(z, ZS_dict):
    fig, axes = plt.subplots(2,1, sharex=True)
    for i in range(len(ZS_dict['ideal_B_list'])):
        ideal_B = ZS_dict['ideal_B_list'][i]
        exp_B = ZS_dict['exp_B_list'][i]
        adia_dev = ZS_dict['adia_dev_list'][i]
        col = COLORS[i]
        residual = exp_B-ideal_B
        axes[0].plot(z, residual, linestyle='dashed', color=col)
        axes[1].plot(z[ZS_START_IND:ZS_END_IND], adia_dev/(A_MAX), color=col)
    axes[0].set_ylabel("Deviation of B Field (G)")
    axes[1].set_ylabel('adiabat_cond.')
    axes[1].set_ylim(0,2)
    axes[1].axhline(1, lw=1, ls='--', color='k')
    axes[1].set_xlabel("Distance down slower (m)")
    return None

def vel_prop_plot(vis, v_dict):
    delta=v_dict['detuning']
    i=v_dict['I']
    fig, axes = plt.subplots(1,1)
    axes.set_xlim(0,0.8)
    axes.set_ylim(0,800)
    axes.axhline(120, ls='--', lw=1, color='k')
    axes.set_title(f'Velocity traj. in ZS (delta = {delta} MHz, I = {i:.2f} Isat)')
    axes.set_ylabel('Velocity (m/s)')
    axes.set_xlabel('Distance down slower (m)')
    for vi in vis:
        v = str(vi)
        axes.plot(v_dict[v]['z'], v_dict[v]['v'], lw=1)
    return None

#B_field_dict = find_ZS_profiles([[0,0,0,0] for i in DETUNINGS], Z_AXIS, DETUNINGS)







