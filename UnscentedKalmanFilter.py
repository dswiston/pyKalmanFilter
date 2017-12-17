#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:36:10 2017

@author: dave
"""

import numpy as np
import sympy as sp
import logging


class UnscentedKalmanFilter:
  
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
  
  likelihood = np.array([])
  
  logger = logging.getLogger()
  handler = logging.StreamHandler()
  formatter = logging.Formatter('%(message)s')
  handler.setFormatter(formatter)
  logger.addHandler(handler)


  def __init__(self,A,R,H,P,B):
    n = np.size(A,0)
    m = np.size(R,0)

    self.A = A
    self.R = R
    self.H = H
    self.P = P
    self.B = B

    self.x = np.array(np.zeros(n))
    self.K = np.array(np.zeros((n,m)))
    self.c = []
    self.mW = [];
    self.cW = [];
    self.likelihood = np.array(np.zeros(n))


  # At each iteration a measurement z and a control input u are provided
  def run(self,z,u):
    
    # Create the sigma points, these are a cloud of points around the current
    # state that are used to "sample" the mean and covariance of the 
    # transformed distribution
    self.xSigmaPts = self.CreateSigmaPts(self.x,self.P,self.c)
    self.logger.info(" Original state estimate: %r", self.x)
    self.logger.info(" Sigma points: %r", self.xSigmaPts)
    
    # -------------------------------------------------------------------------
    # Time updates
    # -------------------------------------------------------------------------
    # Perform the unscented transformation on the predicted states (sigmas)
    [xP,pP,transXSigmaPts,transXSigmaPtsDiff] = \
      self.UnscentedTransform(self.a,self.xSigmaPts,self.mW,self.cW, \
                              np.zeros(len(self.x)))
    self.logger.info(" State prediction: %r", xP)
    self.logger.info(" State covariance prediction: %r", pP)
    self.logger.info(" Transformed state sigma points: %r", transXSigmaPts)
    
    # Perform the unscented transformation on the predicted measurements
    [transMeasEst,transMeasCov,transMeasPts,transMeasDiff] = \
      self.UnscentedTransform(self.h,transXSigmaPts,self.mW,self.cW,self.R)
    self.logger.info(" Measurement estimate: %r", transMeasEst)
    self.logger.info(" Measurement estimate covariance: %r", transMeasCov)
    self.logger.info(" Transformed state sigma points: %r", transMeasPts)
    
    # -------------------------------------------------------------------------
    # Measurement updates
    #--------------------------------------------------------------------------
    # Calculate the innovation (residual of measurement vs state estimated measurement)
    innov = z - transMeasEst
    self.logger.info(" Innovation: %r", innov)
    # Calculate the transformed cross-covariance between the state and the 
    # measurements
    transCrossCov = np.dot(transXSigmaPtsDiff, \
                           np.dot(np.diag(self.cW), np.transpose(transMeasDiff)))
    
    # Update the filter gain
    self.K = np.dot(transCrossCov, np.linalg.inv(transMeasCov))
    self.logger.info(" Kalman filter gain: %r", self.K)
    
    # Make a new state prediction
    self.x = xP + np.dot(self.K, innov)
    self.logger.info(" Updated state estimate: %r", self.x)
    
    # Update the state covariance matrix
    self.P = pP - np.dot(self.K, np.transpose(transCrossCov))
    self.logger.info(" Updated covariance estimate: %r", self.P)
    
    # -------------------------------------------------------------------------
    # Kalman filter stability scoring
    # -------------------------------------------------------------------------
    # This is not part of the filter itself but used to estimate the stability 
    # of the filter
    innovCov = transMeasCov
    # Kalman filter stability scoring
    self.likelihood = np.exp(-0.5 * np.transpose(innov) * \
                      np.dot(np.linalg.inv(innovCov), innov)) / \
                      (np.sqrt((2*np.pi)**3 * np.linalg.det(innovCov)))


  def initialize(self,initStates,measEq,stateEq,vars,debug):
    
    if debug == 1:
      self.logger.setLevel(logging.INFO)
    else:
      self.logger.setLevel(logging.CRITICAL)

    self.x = initStates
    
    # Handle the case where there is only one equation
    if type(measEq) is not list and type(measEq) is not tuple:
      measEq = (measEq,)
    
    # Take the equation expressions and translate them into python functions 
    # using the numpy mathematical library.
    self.h = [0] * len(measEq)
    for eqNdx in range(0,len(measEq)):
      self.h[eqNdx] = sp.lambdify(vars, measEq[eqNdx], modules='numpy')
    
    # Handle the case where there is only one equation
    if type(stateEq) is not list and type(stateEq) is not tuple:
      stateEq = (stateEq,)
    
    # Take the equation expressions and translate them into python functions 
    # using the numpy mathematical library.
    self.a = [0] * len(stateEq)
    for eqNdx in range(0,len(stateEq)):
      self.a[eqNdx] = sp.lambdify(vars, stateEq[eqNdx], modules='numpy')
      
    # Record the number of states and measurements for easy access later
    if hasattr(initStates,'__len__'):
      numStates = len(initStates)
    else:
      numStates = 1
    
    # Create some variables for the UKF.  These variables are tunable but the
    # defaults work well for gaussian distributions
    alpha = 1e-3
    ki = 0
    beta = 2
    
    # Use the defined variables to calculate weights and scaling factors
    lam = alpha**2 * (numStates+ki) - numStates
    c = numStates + lam
    self.mW = np.hstack((lam/c, 0.5/c+np.zeros(2*numStates)))
    self.cW = self.mW
    self.cW[0] = self.cW[0] + (1-alpha**2+beta)
    self.c = np.sqrt(c)


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


  def CreateSigmaPts(self,state,cov,c):
  # Inputs:
  #          x: reference point
  #          P: covariance
  #          c: coefficient
  #
  # Output:
  #          X: sigma points
    
    if hasattr(state,'__len__'):
      numStates = len(state)
    else:
      numStates = 1;
    
    # Choleski decomp is equivalent to taking the sqrt of a matrix
    A = c * np.transpose(np.linalg.cholesky(cov))
    Y = np.tile(state,(1,numStates))
    X = np.hstack((state,Y+A,Y-A))
    return X
    
    
  def UnscentedTransform(self,func,sigmaPts,mW,cW,addCov):
  # Input:
  #  sigmaPts: sigma points to be transformed
  #        Wm: weights for mean
  #        Wc: weights for covariance
  #         R: additive process covariance
  #
  # Output:
  #         y: transformed mean
  #         Y: transformed sampling points
  #     diffY: transformed sampling point deviations from the transformed mean
  #         P: transformed covariance
  
    shape = sigmaPts.shape
    if len(shape) == 1:
      numSigmas = shape
      numVars = 1
    else:
      numSigmas = shape[1]
      numVars = shape[0]
    
    y = np.zeros((numVars,1))
    Y = np.zeros((numVars,numSigmas))
    
    # Loop over each sigma point, calculate the resulting sigma point's 
    # prediction via the non-linear function and add the # weighted 
    # contribution of the sigma point's mean to the resulting transformed mean.
    for ndx in range(0,numSigmas):
      for eqNdx in range(0,numVars):
        # Perform non-linear transformation of the sigma points
        Y[eqNdx,ndx] = func[eqNdx](*sigmaPts[:,ndx])
      # Add weighted contribution of the transformed sigma to the transformed
      # mean
      y = y + mW[ndx] * Y[:,[ndx]]
    # Currently a hack!! <------------- Need to fix/understand why
    y = y / 4

    # Create the transformed different matrix, this quantifies how different
    # each sigma point's measurement is from the transformed mean
    diffY = Y - np.tile(y,(1,numSigmas))
    
    # Create the transformed covariance estimate from a weighted sum of the 
    # sigma point's difference matrix
    P = np.dot(diffY, np.dot(np.diag(cW), np.transpose(diffY))) + addCov
    
    return y,P,Y,diffY