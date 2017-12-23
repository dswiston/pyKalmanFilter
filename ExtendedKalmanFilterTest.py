#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:53:52 2017

@author: dave
"""

from ExtendedKalmanFilter import ExtendedKalmanFilter
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Polar to cartesian estimation
#  ----------------------------------------------------------------------------

# Setup the simulation
numIter = 10000
r = 1
rStd = 1
ang = np.pi/2
angStd = 0.1
x = r*np.cos(ang)
y = r*np.sin(ang)
rMeas = np.random.normal(r,rStd,numIter)
angMeas = np.random.normal(ang,angStd,numIter)
meas = np.vstack((rMeas,angMeas))

# Create the Kalman filter tracker
A = np.eye(2)                             # State difference equation
R = np.eye(2)                             # Measurement covariance
R[0,0] = rStd**2
R[1,1] = angStd**2
h = ['(x**2+y**2)**0.5','atan2(y,x)']     # Measurement to state equations
vars = ['x','y']                          # Variables used in the equations
P = np.eye(2)                             # State estimate covariance
                                          # This is a dummy variable to be 
                                          # calculated later
B = np.eye(2)                             # State control equation

# Create the filter object
kalman = ExtendedKalmanFilter(A,R,P,B)

# Determine initial states
r0 = meas[0,0]
x0 = r0*np.cos(meas[1,0])
y0 = r0*np.sin(meas[1,0])

# Initialize the filter object
kalman.initialize([[x0],[y0]],h,vars,0)

# Calculate the initial state covariance
H0 = kalman.EvalJacobian(kalman.H,np.array([[x0],[y0]]))
P0 = np.dot(np.linalg.inv(H0), np.dot(R, np.transpose(np.linalg.inv(H0))))
kalman.P = P0

# Execute the simulation
stateEstHist = np.zeros((2,numIter))
for noiseNdx in range(0,numIter):
  kalman.run(meas[:,[noiseNdx]],np.zeros((2,1)))
  stateEstHist[:,noiseNdx] = kalman.x[:,0]

# Plot results
plt.plot( meas[0,:] * np.cos(meas[1,:]) )
plt.plot( meas[0,:] * np.sin(meas[1,:]) )
plt.plot(stateEstHist.T)


# -----------------------------------------------------------------------------
# 3d location estimation
# -----------------------------------------------------------------------------

# Setup the simulation
numIter = 10000
r = 10000
rStd = 10                                          # 10m range STD
az = np.pi/2
azStd = np.pi/32                                   # ~5.7deg az STD
el = np.pi/4
elStd = np.pi/32                                   # ~5.7deg el STD
x = r*np.cos(az)*np.cos(el)
y = r*np.sin(az)*np.cos(el)
z = r*np.sin(el)
rangeMeas = np.random.normal(r,10,numIter)
azMeas = np.random.normal(az,np.pi/32,numIter)
elMeas = np.random.normal(el,np.pi/32,numIter)
meas = np.vstack((rangeMeas,azMeas,elMeas))

# Create the Kalman filter tracker
A = np.eye(3)                                     # State difference equation
R = np.eye(3)                                     # Measurement covariance
R[0,0] = rStd**2
R[1,1] = azStd**2
R[2,2] = elStd**2
h = ['(x**2+y**2+z**2)**0.5',\
     'atan2(y,x)','atan2(z,(x**2 + y**2)**0.5)']  # Measurement to state equations
vars = ['x','y','z']                              # Variables used in the equations
P = np.eye(3)                                     # State estimate covariance
                                                  # This is a dummy variable to be 
                                                  # calculated later
B = np.eye(3)                                     # State control equation

# Create the filter object
kalman = ExtendedKalmanFilter(A,R,P,B)

# Determine initial states
r0 = meas[0,0]
x0 = r0*np.cos(meas[1,0])*np.cos(meas[2,0])
y0 = r0*np.sin(meas[1,0])*np.cos(meas[2,0])
z0 = r0*np.sin(meas[2,0])

# Initialize the filter object
kalman.initialize([[x0],[y0],[z0]],h,vars,0)

# Calculate the initial state covariance
H0 = kalman.EvalJacobian(kalman.H,np.array([[x0],[y0],[z0]]))
P0 = np.dot(np.linalg.inv(H0), np.dot(R, np.transpose(np.linalg.inv(H0))))
kalman.P = P0

# Execute the simulation
kalmanState = np.zeros((3,numIter))
for noiseNdx in range(0,numIter):
  kalman.run(meas[:,[noiseNdx]],[[0],[0],[0]])
  kalmanState[:,noiseNdx] = kalman.x.T

plt.plot(meas[0,:]*np.cos(meas[1,:])*np.cos(meas[2,:]))
plt.plot(meas[0,:]*np.sin(meas[1,:])*np.cos(meas[2,:]))
plt.plot(meas[0,:]*np.sin(meas[2,:]))
plt.plot(kalmanState.T)

