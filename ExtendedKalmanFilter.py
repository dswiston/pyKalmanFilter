#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:36:10 2017

@author: dave
"""

import numpy as np
import sympy as sp
import logging


class ExtendedKalmanFilter:
  
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
  h = []   # Non-linear equation relating the state to the measurements
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
    
    # Evaluate/calculate the measurement function and its Jacobian matrix at the predicted state
    hEval = self.EvalFunc(self.h,xP)
    HEval = self.EvalJacobian(self.H,xP)
    self.logger.info(" Function evaluation: %r", hEval)
    self.logger.info(" Jacobian evaluation: %r", HEval)

    # Perform measurement updates
    # Calculate the innovation (residual of measurement vs state estimated measurement)
    innov = z - hEval
    self.logger.info(" Innovation: %r", innov)
    # Calculate the innovation covariance (residual covariance)
    #    - It is a sum of the covariance of the two variables above, the measurement covariance and the 
    #    state estimated measurement covariance.
    #    - First must transform the state covariance to the state estimated measurement covariance
    #      kalman.H(:,:,end) * pP * kalman.H(:,:,end).'
    #    - Then add the measurement covariance (user specified variable)
    innovCov = np.dot(HEval, np.dot(pP, np.transpose(HEval))) + self.R
    self.logger.info(" Innovation covariance: %r", innovCov)
    
    # Update the filter gain
    self.K = np.dot(pP, np.dot(np.transpose(HEval), np.linalg.inv(innovCov)))
    self.logger.info(" Filter gain: %r", self.K)
    
    # Make a new state prediction
    self.x = xP + np.dot(self.K, innov)
    self.logger.info(" Updated state estimate: %r", self.x)
    
    # Update the state covariance matrix
    I = np.eye(self.P.shape[0],self.P.shape[1])
    self.P = np.dot(I - np.dot(self.K, HEval), pP)
    self.logger.info(" Updated state covariance estimate: %r", self.P)
    
    # Kalman filter stability scoring
    self.likelihood = np.exp(-0.5 * np.dot(np.transpose(innov), \
                      np.dot(np.linalg.inv(innovCov), innov))) / \
                      (np.sqrt((2*np.pi)**3 * np.linalg.det(innovCov)))


  def initialize(self,initStates,eq,vars,debug):
    
    if debug == 1:
      self.logger.setLevel(logging.INFO)
    else:
      self.logger.setLevel(logging.CRITICAL)

    self.x = initStates
    
    # Handle the case where there is only one equation
    if type(eq) is not list and type(eq) is not tuple:
      eq = (eq,)
    
    # Take the equation expressions and translate them into python functions 
    # using the numpy mathematical library.
    self.h = [0] * len(eq)
    for eqNdx in range(0,len(eq)):
      self.h[eqNdx] = sp.lambdify(vars, eq[eqNdx], modules='numpy')
    
    # Build an empty Jacobian
    self.H = [[0] * len(vars) for i in range(len(eq))]
    # Also store string representations of the Jacobian result for debug and
    # inspection purposes.
    self.eqH = [[0] * len(vars) for i in range(len(eq))]
    # Loop through each equation and create the partial derivatives
    for eqNdx in range(0,len(eq)):
      for varNdx in range(0,len(vars)):
        self.eqH[eqNdx][varNdx] = sp.diff(eq[eqNdx],vars[varNdx])
        self.H[eqNdx][varNdx] = sp.lambdify(vars, self.eqH[eqNdx][varNdx], \
                           modules='numpy')
    self.logger.info(" Jacobian equations: %r", self.eqH)


  def EvalJacobian(self,jacobian,inputs):
    
    # Find the size of the jacobian matrix
    rows = len(jacobian)
    cols = len(jacobian[0])
    
    # Loop over each equation of the jacobian and evaluate
    jacobEval = [[0.] * cols for i in range(rows)]
    for row in range(0,rows):
      for col in range(0,cols):
        # Evaluate the jacobian functions
        # Use splat operator to split the input list into individual input 
        # params to the symbolicly defined jacobian function
        tmp = jacobian[row][col](*inputs)
        if hasattr(tmp, "__len__"):
          jacobEval[row][col] = tmp[0]
        else:
          jacobEval[row][col] = tmp

    return jacobEval


  def EvalFunc(self,func,inputs):
    
    rows = len(func)
    
    funcEval = np.array([[0.]] * rows)
    for row in range(0,rows):
      # Evaluate the functions
      # Use the splat operator to split the input list into individual input
      # params to the symbolically defined equation
      tmp = func[row](*inputs)
      if hasattr(tmp, "__len__"):
        funcEval[row][0] = tmp[0]
      else:
        funcEval[row][0] = tmp

    return funcEval


