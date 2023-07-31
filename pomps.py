

import jax
import itertools
import numpy as onp
import jax.numpy as np
import ipywidgets as widgets

from jax.numpy.linalg import inv, pinv
from scipy.linalg import solve_discrete_are as dare
from jax import jit, grad
from IPython import display
from toolz.dicttoolz import valmap, itemmap
from itertools import chain

from tqdm import tqdm
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels



def sigmoid(x):
    return 1/(1+np.exp(-x))

def logit(x):
    return np.log(x/(1-x))


def get_thetas(theta):
    gamma = np.exp(theta[0]) #rate at which I recovers
    m = np.exp(theta[1]) #probability of death from cholera
    rho = np.exp(theta[2]) #1/rho is mean duration of short-term immunity
    epsilon = np.exp(theta[3]) # 1/eps is mean duration of immunity
    omega = np.exp(theta[4]) #mean foi
    c = sigmoid(theta[5])/5 #probability exposure infects
    beta_trend = theta[6] / 1000 #trend in foi
    sigma = theta[7]**2 / 2 #stdev of foi perturbations
    tau = theta[8]**2 / 5 #stdev of gaussian measurements
    bs = theta[9:] #seasonality coefficients
    k = 3# 1/(np.exp(theta[3])**2) #1/sqrt(k) is coefficient of variation of immune period
    delta = 0.02 #death rate
    return gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, k, delta

def transform_thetas(gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs):
    return np.concatenate([np.array([np.log(gamma), np.log(m), np.log(rho), np.log(epsilon), np.log(omega),
                    logit(c)*5, beta_trend * 1000, np.sqrt(sigma) * 2, np.sqrt(tau) * 5]), bs])


def rinit(theta, J, covars):
    S_0, I_0, Y_0, R1_0, R2_0, R3_0 = 0.621, 0.378, 0, 0.000843, 0.000972, 1.16e-07
    pop = covars[0,2]
    S = pop*S_0
    I = pop*I_0
    Y = pop*Y_0
    R1 = pop*R1_0
    R2 = pop*R2_0
    R3 = pop*R3_0
    Mn = 0
    return np.tile(np.array([S,I,Y,Mn,R1,R2,R3]), (J,1))

def rinits(thetas, J, covars):
    return rinit(thetas[0], len(thetas), covars)

def dmeas(y, preds, theta, keys=None):
    Mn = preds[3]
    tau = np.exp(theta[8]) #stdev of gaussian measurements
    return jax.scipy.stats.norm.logpdf(y, loc=Mn, scale=tau*Mn)
    #return np.nan_to_num(jax.scipy.stats.norm.logpdf(y, loc=Mn, scale=tau*Mn), nan=-1.0e50)

dmeasure = jax.vmap(dmeas, (None,0,None))
dmeasures = jax.vmap(dmeas, (None,0,0))




'''
def rproc(state, theta, key, covar):
    S, I, Y, Mn, Rs = state[0], state[1], state[2], state[3], state[4:]
    trend, dH, H, seas = covar[0], covar[1], covar[2], covar[3:]
    gamma, m, rho, epsilon, omega, c, beta_trend, sigma, tau, bs, k, delta = get_thetas(theta)
    dt = 1/30
    Mn = 0
    for i in range(30):
        subkey, key = jax.random.split(key)
        dw = jax.random.normal(subkey)*onp.sqrt(dt)
        key = subkey
        foi = omega + (np.exp(beta_trend*trend + np.dot(bs, seas)) + sigma*dw/dt)*I/H
        dS = (k*epsilon*Rs[-1] + rho*Y + dH + delta*H - (foi+delta)*S)*dt
        dI = (c*foi*S - (m+gamma+delta)*I)*dt
        dY = ((1-c)*foi*S - (rho+delta)*Y)*dt
        dRs = [(gamma*I - (k*epsilon+delta)*Rs[0])*dt]
        for i in range(1, len(Rs)):
            dRs.append((k*epsilon*Rs[i-1] - (k*epsilon+delta)*Rs[i])*dt)
        S += dS
        I += dI
        Y += dY
        Mn += m*I*dt
        for i in range(len(Rs)):
            Rs = Rs.at[i].add(dRs[i])
    return np.hstack([np.array([S, I, Y, Mn]), Rs])
'''

def rproc(state, theta, key, covar):
    S, I, Y, deaths, pts = state[0], state[1], state[2], state[3], state[4:]
    trend, dpopdt, pop, seas = covar[0], covar[1], covar[2], covar[3:]
    gamma, deltaI, rho, eps, omega, clin, beta_trend, sd_beta, tau, bs, nrstage, delta = get_thetas(theta)
    dt = 1/240
    deaths = 0
    nrstage = 3
    clin = 1 # HARDCODED SEIR
    rho = 0 # HARDCODED INAPPARENT INFECTIONS
    
    neps = eps*nrstage
    rdeaths = np.zeros(nrstage)
    passages = np.zeros(nrstage+1)
        
    for i in range(20):
        subkey, key = jax.random.split(key)
        dw = jax.random.normal(subkey)*onp.sqrt(dt)
        beta = np.exp(beta_trend*trend + np.dot(bs, seas))
        
        effI = I/pop
        births = dpopdt + delta*pop # births
        passages = passages.at[0].set(gamma*I) #recovery
        ideaths = delta*I #natural i deaths
        disease = deltaI*I #disease death
        ydeaths = delta*Y #natural rs deaths
        wanings = rho*Y #loss of immunity
        
        for j in range(nrstage):
            rdeaths = rdeaths.at[j].set(pts[j]*delta) #natural R deaths
            passages = passages.at[j+1].set(pts[j]*neps) # passage to the next immunity class
            
        infections = (omega+(beta+sd_beta*dw/dt)*effI)*S # infection
        sdeaths = delta*S # natural S deaths
        
        S += (births - infections - sdeaths + passages[nrstage] + wanings)*dt
        I += (clin*infections - disease - ideaths - passages[0])*dt
        Y += ((1-clin)*infections - ydeaths - wanings)*dt
        for j in range(nrstage):
            pts = pts.at[j].add((passages[j] - passages[j+1] - rdeaths[j])*dt)
        deaths += disease*dt # cumulative deaths due to disease
        
        S = np.clip(S, a_min=0); I = np.clip(I, a_min=0); Y = np.clip(Y, a_min=0)
        pts = np.clip(pts, a_min=0); deaths = np.clip(deaths, a_min=0)

        
    return np.hstack([np.array([S, I, Y, deaths]), pts])

rprocess = jax.jit(jax.vmap(rproc, (0, None, 0,None)))
rprocesses = jax.jit(jax.vmap(rproc, (0, 0, 0, None)))