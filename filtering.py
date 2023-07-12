

import jax
import itertools
import numpy as onp
import jax.numpy as np
import ipywidgets as widgets

from jax.numpy.linalg import inv, pinv
from scipy.linalg import solve_discrete_are as dare
from jax import jit, grad
from functools import partial
from IPython import display
from toolz.dicttoolz import valmap, itemmap
from itertools import chain

from tqdm.notebook import tqdm
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

import matplotlib.pyplot as plt
plt.style.use('matplotlibrc')

from resampling import *
from pomps import rinit, rprocess, dmeasure, rinits, rprocesses, dmeasures

def resampler(counts, particlesP, norm_weights):
    J = norm_weights.shape[-1]
    counts = resample(norm_weights)
    particlesF = particlesP[counts]
    weights = norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - np.log(J)
    return counts, particlesF, weights

def no_resampler(counts, particlesP, norm_weights):
    return counts, particlesP, norm_weights

def resampler_thetas(counts, particlesP, norm_weights, thetas):
    J = norm_weights.shape[-1]
    counts = resample(norm_weights)
    particlesF = particlesP[counts]
    weights = norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - np.log(J)
    thetas = thetas[counts]
    return counts, particlesF, weights, thetas

def no_resampler_thetas(counts, particlesP, norm_weights, thetas):
    return counts, particlesP, norm_weights, thetas
    
'''
# Resampling condition
if np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights)) > thresh:
    resamples += 1 #tracker
    # Systematic resampling
    counts = resample(norm_weights, J)
    particlesF = particlesP[counts]
    weights = norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - np.log(J)
else:
    particlesF = particlesP
    weights = norm_weights
'''

def pfilter_helper(t, inputs):
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh = inputs
    J = len(particlesF)
    keys = np.array([jax.random.PRNGKey(onp.random.choice(10000)) for j in range(J)])
    # Get prediction particles 
    if covars is not None:
        particlesP = rprocess(particlesF, theta, keys, covars[t])# if t>0 else particlesF
    else:
        particlesP = rprocess(particlesF, theta, keys)

    oddr = np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights))
    # Systematic resampling
    counts, particlesF, weights = jax.lax.cond(oddr > thresh, 
                                               resampler, 
                                               no_resampler, 
                                               counts, particlesP, norm_weights)



    # Multiply weights by measurement model result
    weights += dmeasure(ys[t], particlesP, theta, keys=keys) #shape (Np,)

    # Obtain normalized weights
    norm_weights, loglik_t = normalize_weights(weights)

    # Sum up loglik
    loglik += loglik_t
    #jax.debug.print(loglik, loglik_t)
    return [particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh]
    
@partial(jit, static_argnums=2)
def pfilter(theta, ys, J, covars=None, thresh=100):
    
    particlesF = rinit(theta, J, covars=covars)
    weights = np.log(np.ones(J)/J)
    norm_weights = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    
    loglik = 0
    
    particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh = jax.lax.fori_loop(
                lower=0, upper=len(ys), body_fun=pfilter_helper, 
                 init_val=[particlesF, theta, covars, loglik, norm_weights, counts, ys, thresh])
    
    return -loglik

def perfilter_helper(t, inputs):
    particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh = inputs
    J = len(particlesF)
    keys = np.array([jax.random.PRNGKey(onp.random.choice(10000)) for j in range(J)])
    
    # Perturb parameters
    thetas += sigmas*np.array(onp.random.normal(size=thetas.shape))
    # Get prediction particles 
    if covars is not None:
        particlesP = rprocesses(particlesF, thetas, keys, covars[t])# if t>0 else particlesF
    else:
        particlesP = rprocesses(particlesF, thetas, keys)# if t>0 else particlesF
    
    oddr = np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights))
    # Systematic resampling
    counts, particlesF, weights, thetas = jax.lax.cond(oddr > thresh, 
                                               resampler_thetas, 
                                               no_resampler_thetas, 
                                               counts, particlesP, norm_weights, thetas)
    
    # Multiply weights by measurement model result
    weights += dmeasures(ys[t], particlesP, thetas, keys=keys).squeeze() #shape (Np,)
    
    # Obtain normalized weights
    norm_weights, loglik_t = normalize_weights(weights)
    
    # Sum up loglik
    loglik += loglik_t
    
    return [particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh]


@partial(jit, static_argnums=2)
def perfilter(theta, ys, J, sigmas, covars=None, a=0.9, thresh=100):
    
    loglik = 0
    thetas = theta + sigmas*onp.random.normal(size=(J, theta.shape[-1]))
    particlesF = rinits(thetas, 1, covars=covars)
    weights = np.log(np.ones(J)/J)
    norm_weights = np.log(np.ones(J)/J)
    counts = np.ones(J).astype(int)
    
    particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh = jax.lax.fori_loop(
                lower=0, upper=len(ys), body_fun=perfilter_helper, 
                 init_val=[particlesF, thetas, sigmas, covars, loglik, norm_weights, counts, ys, thresh])
    
    return -loglik, thetas



#PFILTER
'''  
for t in tqdm(range(len(ys))):
    keys = np.array([jax.random.PRNGKey(onp.random.choice(10000)) for j in range(J)])
    # Get prediction particles 
    if covars is not None:
        particlesP = rprocess(particlesF, theta, keys, covars[t])# if t>0 else particlesF
    else:
        particlesP = rprocess(particlesF, theta, keys)

    resamples += 1 #tracker
    oddr = np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights))
    # Systematic resampling
    counts, particlesF, weights = jax.lax.cond(oddr > thresh, 
                                               partial(resampler, J=J), 
                                               partial(no_resampler, J=J), 
                                               counts, particlesP, norm_weights)



    # Multiply weights by measurement model result
    keys = np.array([jax.random.PRNGKey(onp.random.choice(10000)) for j in range(J)])
    weights += dmeasure(ys[t], particlesP, theta, keys=keys) #shape (Np,)

    # Obtain normalized weights
    norm_weights, loglik_t = normalize_weights(weights)

    # Sum up loglik
    loglik += loglik_t
'''

#PERFILTER
'''
# inner filtering loop
for t in tqdm(range(len(ys))):
    keys = np.array([jax.random.PRNGKey(onp.random.choice(10000)) for j in range(J)])

    # Perturb parameters
    thetas += sigmas*np.array(onp.random.normal(size=thetas.shape))
    # Get prediction particles 
    if covars is not None:
        particlesP = rprocesses(particlesF, thetas, keys, covars[t])# if t>0 else particlesF
    else:
        particlesP = rprocesses(particlesF, thetas, keys)# if t>0 else particlesF

    # Resampling condition
    if np.exp(np.max(norm_weights))/np.exp(np.min(norm_weights)) > thresh:
        resamples += 1 #tracker
        # Systematic resampling
        counts = resample(norm_weights, J)
        particlesF = particlesP[counts]
        thetas = thetas[counts]
        weights = norm_weights[counts] - jax.lax.stop_gradient(norm_weights[counts]) - np.log(J)
    else:
        particlesF = particlesP
        weights = norm_weights


    keys = np.array([jax.random.PRNGKey(onp.random.choice(10000)) for j in range(J)])
    # Multiply weights by measurement model result
    weights += dmeasures(ys[t], particlesP, thetas, keys=keys).squeeze() #shape (Np,)

    # Obtain normalized weights
    norm_weights, loglik_t = normalize_weights(weights)

    # Sum up loglik
    loglik += loglik_t
'''
