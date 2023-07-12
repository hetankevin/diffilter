

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

from functools import partial

from resampling import *
from filtering import *
from pomps import *

import matplotlib.pyplot as plt
plt.style.use('matplotlibrc')

MONITORS = 1



@partial(jit, static_argnums=2)
def jgrad(theta_ests, ys, J, covars, thresh):
    return jax.grad(pfilter)(theta_ests, ys, J, covars=covars, thresh=thresh)

@partial(jit, static_argnums=2)
def jvg(theta_ests, ys, J, covars, thresh):
    return jax.value_and_grad(pfilter)(theta_ests, ys, J, covars=covars, thresh=thresh)

@partial(jit, static_argnums=2)
def jhess(theta_ests, ys, J, covars, thresh):
    return jax.hessian(pfilter)(theta_ests, ys, J, covars=covars, thresh=thresh)

# From https://arxiv.org/pdf/1909.01238.pdf
def line_search(obj, curr_obj, pt, grad, direction, k=1, eta=0.9, xi=10, tau = 10, c=0.1, frac=0.5, stoch=False):
    itn = 0
    eta = min([eta, xi/k]) if stoch else eta
    next_obj = obj(pt + eta*direction)
    while next_obj > curr_obj + eta*c*grad.T @ direction or np.isnan(next_obj):
        eta *= frac
        itn += 1
        if itn > tau:
            break
    return eta



#rerun with diff trajs each time
def train(theta_ests, ys, covars=None, J=5000, Jh=1000, method='Newton', itns=20, beta=0.9, eta=0.0025, c=0.1, max_ls_itn=10, thresh=100, verbose=False, scale=False, ls=False): 
    Acopies = []
    grads = []
    hesses = []
    logliks = []
    hess = np.eye(theta_ests.shape[-1])
    
    
    for i in tqdm(range(itns)):
        if MONITORS == 1:
            loglik, grad = jvg(theta_ests, ys, J, covars=covars, thresh=thresh)
        else:
            grad = jgrad(theta_ests, ys, J, covars=covars, thresh=thresh)
            loglik = np.mean(np.array([pfilter(theta_ests, ys, J, covars=covars, thresh=thresh) for i in range(MONITORS)]))
        
        if method=='Newton':
            hess = jhess(theta_ests, ys, Jh, covars=covars, thresh=thresh)
            direction = -np.linalg.pinv(hess) @ grad
            #hess here is hessian
        elif method == 'WeightedNewton':
            if i == 0:
                hess = jhess(theta_ests, ys, Jh, covars=covars, thresh=thresh)
                direction = -np.linalg.pinv(hess) @ grad
            else:
                hess = jhess(theta_ests, ys, Jh, covars=covars, thresh=thresh)
                weights = np.array([((i+1)**(onp.log(i+1))) for i in range(len(hesses))])
                direction = -np.linalg.pinv(
                                       np.einsum('ijk,i->jk', np.array(hesses), weights) 
                                        + ((i+1)**(onp.log(i+1)))*hess) @ grad
            #hess here is hessian, but we update according to (t+1)^log(t+1) weights
        elif method=='BFGS' and i > 1:
            s_k = eta * direction
            y_k = grad - grad[-1]
            rho_k = np.reciprocal(np.dot(y_k, s_k))
            sy_k = s_k[:, np.newaxis] * y_k[np.newaxis, :]
            w = np.eye(theta_ests.shape[-1], dtype=rho_k.dtype) - rho_k * sy_k
            hess = (np.einsum('ij,jk,lk', w, hess, w)
                     + rho_k * s_k[:, np.newaxis] * s_k[np.newaxis, :])
            hess = np.where(np.isfinite(rho_k), hess, hess)
            direction = -hess @ grad #hess here is inverse hessian
        else:
            direction = -grad
            
        Acopies.append(theta_ests)
        logliks.append(loglik)
        grads.append(grad)
        hesses.append(hess)

        if scale:
            direction = direction/np.linalg.norm(direction)
            
        eta = line_search(partial(pfilter, ys=ys, J=J, covars=covars, thresh=thresh), 
                          loglik, theta_ests, grad, direction, k=i+1, eta=beta, c=c, tau=max_ls_itn) if ls else eta

        if i%1==0 and verbose:
            print(theta_ests, eta, logliks[i])

        theta_ests += eta*direction
        
    logliks.append(np.mean(np.array([pfilter(theta_ests, ys, J, covars=covars, thresh=thresh) for i in range(MONITORS)])))
    Acopies.append(theta_ests)
    
    return np.array(logliks), np.array(Acopies)


    
def mif(theta, ys, sigmas, sigmas_init, covars=None, M=10, 
        a=0.9, J=100, thresh=100, monitor=False, verbose=False):
    
    logliks = []
    params = []
    
    thetas = theta + sigmas_init*onp.random.normal(size=(J, theta.shape[-1]))
    params.append(thetas)
    if monitor:
        loglik = np.mean(np.array([pfilter(theta, ys, J, covars=covars, thresh=thresh) 
                                   for i in range(MONITORS)]))
        logliks.append(loglik)
        
    # outer iterative loop 
    for m in tqdm(range(M)):
        # annealing pertubations
        sigmas *= a
        thetas += sigmas*onp.random.normal(size=thetas.shape)
        loglik_ext, thetas = perfilter(thetas, ys, J, sigmas, covars=covars, a=a, thresh=thresh)
        
        params.append(thetas)
        
        #code for monitoring logliks and verbose output
        if monitor:
            loglik = np.mean(np.array([pfilter(thetas.mean(0), ys, J, covars=covars, thresh=thresh) 
                                       for i in range(MONITORS)]))
            logliks.append(loglik)
                  
            if verbose:
                print(loglik)
                print(get_thetas(thetas.mean(0)))
        
    return np.array(logliks), np.array(params)

def newtif(theta, ys, sigmas, sigmas_init, covars=None, M=10,
           a=0.9, beta=0.9, tau=10, J=100, thresh=100, monitor=False, verbose=False):
    
    logliks = []
    params = []
    
    # Pertubation on first iteration
    thetas = theta + sigmas_init*onp.random.normal(size=(J, theta.shape[-1]))
    params.append(thetas)
    
    if monitor:
        loglik = np.mean(np.array([pfilter(thetas.mean(0), ys, J, covars=covars, thresh=thresh) for i in range(MONITORS)]))
        logliks.append(loglik)
        
    jpgrad = jax.jit(jax.grad(perfilter, has_aux=True))
    # outer iterative loop 
    for m in tqdm(range(M)):
        # annealing pertubations
        sigmas *= a
        #thetas += sigmas*onp.random.normal(size=thetas.shape)
        if m == 0:
            pass
        else:
            '''
            # Newton update
            grad = jax.grad(perfilter, has_aux=True)(theta, 
                ys, sigmas=sigmas, a=a, J=J, thresh=thresh)[0]
            hess = jax.hessian(perfilter, has_aux=True)(theta, 
                ys, sigmas=sigmas, a=a, J=J, thresh=thresh)[0]
            direction = -np.linalg.pinv(hess) @ grad
            #eta = line_search(partial(pfilter, ys=ys, J=J, thresh=thresh), 
            #                  loglik, thetas.mean(0), grad, direction, k=m+1, eta=beta, tau=tau)
            thetas += beta*direction
            print(beta*direction, thetas.mean(0))
            '''
            grad = jpgrad(thetas, 
                ys, J, sigmas, covars=covars, a=a,thresh=thresh)[0]
            direction = -np.nan_to_num(grad, 0)
            eta = np.sqrt(len(ys)*sigmas**2)
            thetas += eta*direction
            #print(eta, eta*np.mean(direction, axis=0), thetas.mean(0))
        '''
        loglik_thetas_ext, grad = jax.value_and_grad(
                                    perfilter, 
                                    has_aux=True)(
                            thetas, ys, sigmas=sigmas, 
                            a=a, J=J, thresh=thresh)
        loglik_ext, thetas = loglik_thetas_ext
        '''
        loglik_ext, thetas = perfilter(thetas, ys, J, sigmas, covars=covars, a=a, thresh=thresh)
        theta = thetas.mean(0)
        
        params.append(thetas)
        
        #code for monitoring logliks and verbose output
        if monitor:
            loglik = np.mean(np.array([pfilter(thetas.mean(0), ys, J, covars=covars, thresh=thresh) 
                                       for i in range(MONITORS)]))
            logliks.append(loglik)
                  
            if verbose:
                print(loglik)
                print(get_thetas(thetas.mean(0)))
        
    return np.array(logliks), np.array(params)