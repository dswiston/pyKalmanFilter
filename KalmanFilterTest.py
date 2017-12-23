#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:53:52 2017

@author: dave
"""

from KalmanFilter import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import copy


# -----------------------------------------------------------------------------
# Simple constant value estimation
#  ----------------------------------------------------------------------------

# Create the Kalman filter tracker
A = np.eye(2)                             # State difference equation
R = np.eye(2) * ((100**2,0),(0,1000**2))  # Measurement covariance
H = np.eye(2)                             # Measurement to state equation
P = np.eye(2) * ((100**2,0),(0,1000**2))  # Initial state covariance
B = np.eye(2)                             # State control equation

# Create the filter object
kalman = KalmanFilter(A,R,H,P,B)
# Initialize the filter object
kalman.initialize((0,0),0)

# Setup the simulation
numIter = 10000
meas = np.random.normal([5.0,5.0],[100.0,1000.0],(numIter,2))
kalmanStateTrack = [0] * numIter
kalmanState = np.zeros((2,numIter))

# Execute the simulation
for noiseNdx in range(0,numIter):
  kalman.run(meas[noiseNdx,:],[0,0])
  kalmanStateTrack[noiseNdx] = copy.copy(kalman)
  kalmanState[:,noiseNdx] = kalman.x

# Plot the results
plt.plot(meas)
plt.plot(kalmanState.T)

# -----------------------------------------------------------------------------
# Simple 1d motion estimation
#  ----------------------------------------------------------------------------

# Create the Kalman filter tracker
A = np.array([[1,1],[0,1]])               # State difference equation
R = np.eye(2) * ((100**2,0),(0,1000**2))  # Measurement covariance
H = np.eye(2)                             # Measurement to state equation
P = np.eye(2) * ((100**2,0),(0,1000**2))  # Initial state covariance
B = np.eye(2)                             # State control equation

# Create the filter object
kalman = KalmanFilter(A,R,H,P,B)
# Initialize the filter object
kalman.initialize((0,0),0)

# Setup the simulation
numIter = 1000
rangeMeas = np.arange(0,numIter) + np.random.normal(0,100,numIter)
velMeas = np.zeros(numIter) + np.random.normal(0,1000,numIter)
meas = np.vstack((rangeMeas,velMeas))
meas = meas.T
kalmanStateTrack = [0] * numIter
kalmanState = np.zeros((2,numIter))

# Execute the simulation
for noiseNdx in range(0,numIter):
  kalman.run(meas[noiseNdx,:],[0,0])
  kalmanStateTrack[noiseNdx] = copy.copy(kalman)
  kalmanState[:,noiseNdx] = kalman.x

# Plot the results
plt.plot(meas)
plt.plot(kalmanState.T)
