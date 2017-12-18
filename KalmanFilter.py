#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:36:10 2017

@author: dave
"""

import numpy as np
import logging


class KalmanFilter:
  
  x = []   # Kalman filter predictions.  Updated each filter iteration (step).
           # Size n
  R = []   # Measurement noise covariance matrix.  Generally held as a constant and not updated over
           # time (steps).
           # Size m x m
  A = []   # Difference equation, relates the state at the previous step to the state at the 
           # current step.  This is a constant, the relationship between the current and the next 
           # state does not vary over time (steps).
           # Size n x n
  H = []   # Measurement equation that relates the state to the measurements.  Since the measurement
           # has a non-linear relationship to the state, this is a Jacobian matrix for the function 
           # relating the measurements to the state.  This is updated/calculated at each step.
           # Size m x n
  P = []   # Estimate error (states) covariance matrix.  This is calculated/updated at each step.
           # Size n x n
  K = []   # Kalman filter gain or "blending" factor that minimizes the a-posteriori error 
           # covariance.  This is calculated/updated at each step.
           # Size n x m
  B = []   # This matrix relates the control input to the state.  Movement of the platform will
           # cause the state to change.
           # Size n x l
  
  likelihood = []  # Not part of core algorithm.  Likelihood gives insight into
                   # the stability of the filter over time.
  
  # Define logging so that the user can get debug style feedback when the 
  # filter updates at each step
  logger = logging.getLogger()
  handler = logging.StreamHandler()
  formatter = logging.Formatter('%(message)s')
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  
  
  def __init__(self,A,R,H,P,B):
    self.A = A
    self.R = R
    self.H = H
    self.P = P
    self.B = B


  # At each iteration a measurement z and a control input u are provided
  def run(self,z,u):
    
    # Perform the prediction updates
    xP = np.dot(self.A, self.x) + np.dot(self.B, u)
    pP = np.dot(self.A, np.dot(self.P, np.transpose(self.A)))
    self.logger.info(" State prediction: %r", xP)
    self.logger.info(" State covariance prediction: %r", pP)
    
    # Perform measurement updates
    # Update the filter gain
    self.K = np.dot(pP, np.dot(np.transpose(self.H), np.linalg.inv( np.dot(self.H, np.dot(pP, np.transpose(self.H))) + self.R )))
    self.logger.info(" Filter gain: %r", self.K)

    # Make a new state prediction
    innov = z - np.dot(self.H,xP)
    self.x = xP + np.dot(self.K, innov)
    self.logger.info(" Updated state estimate: %r", self.x)

    # Update the state covariance matrix
    self.P = np.dot(np.eye(np.size(self.P,0),np.size(self.P,1)) - np.dot(self.K, self.H), pP)
    self.logger.info(" Updated state covariance estimate: %r", self.P)

    # Kalman filter stability scoring
    innovCov = np.dot(self.H, np.dot(pP, np.transpose(self.H))) + self.R
    self.likelihood = np.exp(-0.5 * np.dot(np.transpose(innov), \
                      np.dot(np.linalg.inv(innovCov), innov))) / \
                      (np.sqrt((2*np.pi)**3 * np.linalg.det(innovCov)))


  def initialize(self,initStates,debug):
    self.x = initStates
    
    if debug == 1:
      self.logger.setLevel(logging.INFO)
    else:
      self.logger.setLevel(logging.CRITICAL)
      