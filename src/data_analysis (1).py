# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 09:06:09 2022

@author: bensa
"""

import numpy as np # math functions
import scipy # scientific functions
import matplotlib.pyplot as plt # for plotting figures and setting their properties
import pandas as pd # handling data structures (loaded from files)
from scipy.stats import linregress # contains linregress (for linear regression)
from scipy.optimize import curve_fit as cfit # non-linear curve fitting
from sklearn.metrics import r2_score # import function that calculates R^2 score

#%% Potential

def potential(x,y,a,C): #defines the func
    return -C*np.log(np.sqrt((x-a)**2+y**2)/a)+C*np.log(np.sqrt((x+a)**2+y**2)/a) #calculates potential for 1 point

C = 1
a = 1
L = 3
N = 100
coord = np.linspace(-L, L , N) # defines coordinates
coord_x, coord_y = np.meshgrid(coord, coord)

V_xy = potential(coord_x, coord_y, a, C) #calculates potential for all points using the function above

plt.figure() #generates a figure to plot on
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.grid()
plt.pcolormesh(coord_x, coord_y, V_xy) #plots the potential function along our coord system on a color mesh
plt.colorbar() #adds a colorbar

plt.contour(coord_x, coord_y, V_xy, np.linspace(-1,1,9).tolist(), cmap='hot')
#the cmap parameter defines what spectrum of colors to use when coloring in the color map, i.e. different presets like hot, turbo, summer, etc.

x = np.linspace(-1,1,100)
V_x = potential(x, 0, a, C)
plt.figure()
plt.xlabel('x [m]')
plt.ylabel('Potential [V]')
plt.plot(x,V_x,'.', label="calculated potential")
plt.legend()
plt.grid()

#%% Capacitor

def V_decay(t,tau,V0):
    return V0*np.exp(-t/tau)

eps0 = scipy.constants.epsilon_0 # F/m
D = 18e-2 # m
d = 0.5e-3 # m

C_theoretical = eps0*np.pi*D**2/(4*d)
R_tot = 38.4e3 # Ohm
R = 977 # Ohm
tau_theoretical = R_tot*C_theoretical

C_data = pd.read_csv('capacitor.csv')
C_data = C_data.rename(columns = {"time (sec)":"t", "ch2":"V_R"})
C_data["V_C"] = C_data["ch1"] - C_data["V_R"]

t = np.array(C_data['t'].values)
V_C = np.array(C_data['V_C'].values)

plt.figure()
plt.xlabel('Time [s]')
plt.ylabel('V_C [V]')
plt.grid()
plt.plot(t,V_C,'.', label="capacitor voltage")
plt.legend(['capacitor voltage'])

#CURVE FITTING
#Curve fitting involves finding the a function which closely resembles data point.
#It's done by choosing the type of curve (how we expect the data to behave), and
#usually using an automated program to calculate the parameters for the curve
#which yield the smallest residuals (residuals are the vertical distances between
#data and the curve itself)

p_optimal, p_covariance = cfit(V_decay,C_data['t'], C_data["V_C"])

#10.
#cfit finds the tau and V0 which generate the curve of best fit for the data in
#V_C when the function V_decay is used with the dependent variable data from t

#11.
#cfit outputs the optimal values of tau and V0 into the first and second elements
#respectively of p_optimal
tau_fit, V0_fit = p_optimal

#12.
#p_covariance is the estimated cov matrix of tau and V0. The innate error in each
#variable is the standard deviation, or the square root of the covariance with 
#itself, easily derived
tau_error, V0_error = np.sqrt(np.diag(p_covariance))

#13.
plt.plot(C_data['t'], V_decay(C_data['t'],p_optimal[0],p_optimal[1]),label="fitted curve")
plt.legend()
#Just by looking at the graph we can see the fit is bad.

#14.
#p0 allows us to set initial guesses for the parameters, improving the performance
#of the cfit method. We are given the expected value of tau and we set V0 by observing
#that the initial value of the function and data should be the same, V=4
p0 = [tau_theoretical, 4]
p_optimal, p_covariance = cfit(V_decay,C_data['t'], C_data["V_C"], p0)
plt.plot(C_data['t'], V_decay(C_data['t'],p_optimal[0],p_optimal[1]),label="fitted curve with p0")
plt.legend()
#The fit is not perfect but way better than before.

#15.
V_expected = V_decay(C_data['t'],p_optimal[0],p_optimal[1]) 
chi2 = np.sum(((V_expected-V_C)/0.05)**2)
dof = len(V_C)-2

#16.
one_minus_p = scipy.stats.chi2.cdf(chi2, dof)
#--> p~1

#17.
R2 = r2_score(V_C, V_expected)
#R2 = 0.997 which is logical because we saw earlier that the fit was very good.

#18.
plt.figure()
plt.xlabel('Time [s]')
plt.ylabel('log(V_C)')
plt.plot(t, np.log(V_C))
plt.grid()

#19.
t1 = 0
t2 = 0.0000425
inds = (C_data['t'] > t1) & (C_data['t'] < t2)
plt.plot(C_data['t'][inds], np.log(C_data["V_C"])[inds],'.', label="data")

#20.
#Linear regression is a specific case of curve fitting in which the function we
#are trying to fit to the data is linear. Therefore, the parameters are the slope
#and the y intercept 

#21.
reg = linregress(C_data['t'][inds], np.log(C_data["V_C"])[inds])
print(reg)

#22.
#We derive the original parameters by applying ln() to V_decay and equating to
#the equation for the line of best fit, y=mx+b, where m=slope, b=y intercept
tau_reg = -1/reg.slope 
V0_reg = np.e**(reg.intercept) 

#23.
#The error for the original parameters is propagated from the equations above:
#tau_reg_error = (derivative of tau WRT m)*(m error)
#V0_reg_error = (derivative of V0 WRT b)*(b error)
tau_reg_error = reg.stderr/reg.slope**2
V0_reg_error = reg.intercept_stderr*np.e**(reg.intercept) 

#24.
R2_reg = reg.rvalue**2

#25.
plt.plot(t[inds], t[inds]*reg.slope+reg.intercept)

#26.
C_data["int_V_R"] = scipy.integrate.cumtrapz(C_data["V_R"], x = t, initial = 0)

#27
plt.figure()
plt.xlabel('Integral of V_R [Vs]')
plt.ylabel('Change in V_C [V]')
plt.plot(C_data["int_V_R"], C_data["V_C"]-C_data["V_C"][0], label = "V_C as function of integral of V_R")
#The data is linear throughout the entire domain

#28.
reg2 = scipy.stats.linregress(C_data["int_V_R"], C_data["V_C"]-C_data["V_C"][0])
C_meas = 1/(R*reg2.slope)

#29.
plt.plot(C_data["int_V_R"], C_data["int_V_R"]*reg2.slope+reg2.intercept, label = 'regression')
plt.legend()
plt.grid()

#%% Ohm

def I_R(V2, R1):
    return V2/R1
    
def V_R(V1, V2):
    return V1-V2
    
def R_t(V_R, I_R):
    return V_R/I_R
    
def P_t(V_R, I_R):
    return V_R*I_R
    
def Energy(P_t, t):
    return scipy.integrate.cumtrapz(P_t, x=t, initial=0)

R1 = 5.48 #Ohm

R_data=pd.read_csv('ohm.csv', header=1, usecols=[3, 4, 5, 7, 8])
#header specifies the row number in which column names are located

R_data = R_data.rename(columns = {"Time (s)":"t", "1 (VOLT)":"V1", "2 (VOLT)":"V2"})

V = V_R(R_data["V1"], R_data["V2"])
I = I_R(R_data["V2"], R1)

R = R_t(V, I)
E = Energy(P_t(V, I), R_data["t"])

plt.figure()
plt.grid()
plt.plot(E, R)
plt.xlabel("Energy [J]")
plt.ylabel("Resistance [Ohm]")

inds = (E > 0.05) & (E < 0.95)
reg3 = scipy.stats.linregress(E[inds], R[inds])
plt.plot(E[inds], E[inds]*reg3.slope+reg3.intercept)

R0 = reg3.intercept 
alpha_C_heat = reg3.slope/R0 
#The results seem correct.

#%% Inductance

#6.
def flux(voltage, time):
   return scipy.integrate.cumtrapz(voltage, x=time, initial = 0)

#2.
h = np.array([0.3, 0.24, 0.18, 0.14, 0.08]) #m

#3-4.
Ind_data = []
for n in range(0,5):
    df = pd.read_csv('Trace %d.csv'%n, header = 1, usecols=[3, 4, 5])
    df = df.rename(columns = {"Time (s)":"t", "1 (VOLT)":"ref", "2 (VOLT)":"signal"})
    Ind_data.append(df)

#5.
plt.figure()
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Voltage [V]")
for df in Ind_data:
    plt.plot(df["t"], df["ref"])
    plt.plot(df["t"], df["signal"])

#7-11.
t_coil = []
plt.figure()
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Flux")
for df in Ind_data:
    f_ref =     flux(df["ref"], df["t"])
    ind_max_ref = np.argmax(np.abs(f_ref))
    df["t"] -= df["t"][ind_max_ref]
    f_signal = flux(df["signal"], df["t"])
    ind_max_signal = np.argmax(np.abs(f_signal))
    t_coil.append(df["t"][ind_max_signal])
    plt.plot(df["t"], f_ref)
    plt.plot(df["t"], f_signal)
    plt.plot(df["t"][ind_max_ref], f_ref[ind_max_ref],'*')
    plt.plot(df["t"][ind_max_signal], f_signal[ind_max_signal],'*')
# The points are correct.

#12.
y = h/t_coil
plt.figure()
plt.grid()
plt.plot(t_coil, y,'.')
plt.xlabel("t_coil [s]")
plt.ylabel("h/t_coil [ms^-1]")

#13.
h_error = 0.001
t_coil_error = 0.002
#To calculate propagated error, same method as exercise above:
#y_error^2 = [(derivative of y WRT h)*(h error)]^2+[(derivative of y WRT t_coil)*(t_coil error)]^2
y_error = 0.0425
plt.errorbar(t_coil, y, y_error, t_coil_error, ls = 'none')

#14.
reg4 = scipy.stats.linregress(t_coil, y)
v_0 = reg4.intercept
a = 2*reg4.slope
#a is close enough to g=9.81ms^-2

#15.
y_reg = np.array(t_coil)*reg4.slope+reg4.intercept
plt.plot(np.array(t_coil), y_reg)

#16.
R2_reg4 = reg4.rvalue**2

#17.
chi2_coil = np.sum(((y_reg-y)/y_error)**2)
dof_coil = len(y)-2
one_minus_p_coil = scipy.stats.chi2.cdf(chi2_coil, dof_coil)
print("p = 0.915")

#%% Notes to self
"""
Useful functions
-All of plt
-np.argmax
-cfit
-scipy.stats.linregress
-.append
-data frames
