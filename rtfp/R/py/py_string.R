## This is a standalone R function which call python and uses data objects from
## inside R. This function will go into package but is currently separate to make
## testing easier

library(rstanarm)
data(roaches)
roaches$roach1<-roaches$roach1/100;# manual

glm_negbin<-function(thedata=NULL) {
  data_l=thedata # local copy inside frame otherwise python cant find it
  assign("data_l", data_l, envir = .GlobalEnv)

  #py$data<-thedata # this is needed as explicitly passes argument into py

  bigstring<-r"(

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
tfb = tfp.bijectors
import numpy as np
import pandas as pd
import time

data=r.data_l
y_data=tf.convert_to_tensor(data.iloc[:,0], dtype = tf.float32)
roach_data=tf.convert_to_tensor(data.iloc[:,1], dtype = tf.float32)
trt_data=tf.convert_to_tensor(data.iloc[:,2], dtype = tf.float32)
snr_data=tf.convert_to_tensor(data.iloc[:,3], dtype = tf.float32)
exposure_data=tf.convert_to_tensor(data.iloc[:,4], dtype = tf.float32)

def make_observed_dist(phi, beta_senior,beta_treatment, beta_roach,alpha):
    """Function to create the observed Normal distribution."""
    alpha_expanded = tf.expand_dims(alpha, -1)  # (3, 1)
    beta_roach_expanded = tf.expand_dims(beta_roach, -1)    # (3, 1)
    beta_treatment_expanded = tf.expand_dims(beta_treatment, -1)    # (3, 1)
    beta_senior_expanded = tf.expand_dims(beta_senior, -1)    # (3, 1)
    beta_expos_expanded = tf.expand_dims(1.0, -1)

    logmu = alpha_expanded + beta_roach_expanded * roach_data + beta_treatment_expanded * trt_data + beta_senior_expanded * snr_data +tf.math.log(exposure_data)

    phi_expanded = tf.expand_dims(phi, -1)  # (3, 1)
    mu = tf.math.exp(logmu)
    #r = tf$expand_dims(1.0, -1L)/ phi_expanded;  # total_count = 5
    #probs <- r / (r + mu)

    prob = phi_expanded/(phi_expanded+mu)
    #prob<-(phi_expanded)/(mu+phi_expanded)
    # Create distribution for observations
    return(tfd.Independent(
        #tfd_normal(loc = mu, scale = sigma_expanded),
        #mu = exp(mu)
        #phi <- 0.2  # scale/overdispersion

        #r = tf$expand_dims(1.0, -1L)/ sigma_expanded;  # total_count = 5
        #probs <- r / (r + mu)
        tfd.NegativeBinomial(total_count = phi_expanded, probs = 1-prob),
        reinterpreted_batch_ndims = 1
    ))



# APPROACH 1. Define the joint distribution without matrix mult
model = tfd.JointDistributionSequential([
  tfd.Normal(loc=0., scale=5., name="alpha"),  # # Intercept (alpha)
  tfd.Normal(loc=0., scale=2.5, name="beta_roach"),  # # Slope (beta_roach1)
  tfd.Normal(loc=0., scale=2.5, name="beta_treatment"),  # # Intercept (alpha)
  tfd.Normal(loc=0., scale=2.5, name="beta_senior"),  # # Slope (beta_roach1)
  tfd.Exponential(rate=1., name="phi"),
  make_observed_dist
])

# Approach 1.
tf.random.set_seed(9999)
#a=model.sample()
#print(model)
#a=model.sample()

def log_prob_fn(alpha, beta_roach,beta_treatment,beta_senior,phi):
  """Unnormalized target density as a function of states."""
  return model.log_prob((
      alpha, beta_roach,beta_treatment,beta_senior,phi, y_data))

def neg_log_prob_fn(pars):
    alpha=pars[[0]]
    beta_roach=pars[[1]]
    beta_treatment=pars[[2]]
    beta_senior=pars[[3]]
    phi=pars[[4]]
    """Unnormalized target density as a function of states."""
    return -model.log_prob((
      alpha, beta_roach,beta_treatment,beta_senior,phi, y_data))


print(log_prob_fn(0.1,0.2,0.3,0.5,0.1))
start = tf.constant([0.1,0.2,0.3,0.5,0.1],dtype = tf.float32)
print(neg_log_prob_fn(start))

#### get starting values by find MLE
if(True):
    start = tf.constant([0.1,0.2,0.3,0.5,0.1],dtype = tf.float32)  # Starting point for the search.
    optim_results = tfp.optimizer.nelder_mead_minimize(neg_log_prob_fn,
                 initial_vertex=start, func_tolerance=1e-04,max_iterations=1000)

    print(optim_results.initial_objective_values)
    print(optim_results.objective_value)
    print(optim_results.position)

# bijector to map contrained parameters to real
unconstraining_bijectors = [
    tfb.Identity(),
    tfb.Identity(),
    tfb.Identity(),
    tfb.Identity(),
    tfb.Exp()
]

num_results=1000
num_burnin_steps=1000

sampler = tfp.mcmc.TransformedTransitionKernel(
    tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=log_prob_fn,
        step_size=tf.cast(0.5, tf.float32)), #tf.cast(0.1, tf.float32)),
    bijector=unconstraining_bijectors
    )

adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=sampler,
    num_adaptation_steps=int(0.8 * num_burnin_steps),
    target_accept_prob=tf.cast(0.75, tf.float32))


istate = optim_results.position
#print(initial_state)

print("here initial_state")
#print(initial_state)

#current_state=initial_state

#current_state=[tf.constant([2.8],dtype = tf.float32),
#               tf.constant([1.3],dtype = tf.float32),
#               tf.constant([-0.76],dtype = tf.float32),
#               tf.constant([-0.33],dtype = tf.float32),
#               tf.constant([0.27],dtype = tf.float32)]

#current_state=[tf.constant([initial_state[[0]].numpy()],dtype = tf.float32),
#               tf.constant([initial_state[[1]].numpy()],dtype = tf.float32),
#               tf.constant([initial_state[[2]].numpy()],dtype = tf.float32),
#               tf.constant([initial_state[[3]].numpy()],dtype = tf.float32),
#               tf.constant([initial_state[[4]].numpy()],dtype = tf.float32)]

n_chains=4
current_state = [tf.expand_dims(tf.repeat(istate[0],repeats=n_chains,axis=-1),axis=-1),
                 tf.expand_dims(tf.repeat(istate[1],repeats=n_chains,axis=-1),axis=-1),
                 tf.expand_dims(tf.repeat(istate[2],repeats=n_chains,axis=-1),axis=-1),
                 tf.expand_dims(tf.repeat(istate[3],repeats=n_chains,axis=-1),axis=-1),
                 tf.expand_dims(tf.repeat(istate[4],repeats=n_chains,axis=-1),axis=-1)
                 ]
print("current state")
print(current_state)
# 3. Add a new dimension and then repeat
# First, reshape from (3,) to (3, 1)
#input_expanded = tf.expand_dims(initial_state, axis=-1)
#print(f"Expanded Shape: {input_expanded.shape}\n") # Shape is (3, 1)

# Now, repeat the values 3 times along the new last axis (axis=-1)
#output_matrix = tf.repeat(input_expanded, repeats=1, axis=-1)

#print("matrix")
#print(output_matrix)
#exit()





#a=tf.constant([-0.01205934,  2.8705761, -0.5943442, 0.59550726, 0.0756483], dtype=tf.float32)

#print("here current_state")
#print(current_state)
#print("here current_state2")
#print(current_state2)
#exit()

#print(current_state)

#initial_state = [tf.cast(x, tf.float32) for x in [1., 1., 1., 1., 1.]]

#tfp.mcmc.sample_chain(
#      kernel=adaptive_sampler,
#      current_state=current_state,
#      num_results=num_results,
#      num_burnin_steps=num_burnin_steps)

# Speed up sampling by tracing with `tf.function`.
@tf.function(autograph=False, jit_compile=False)
def do_sampling():
  return tfp.mcmc.sample_chain(
      kernel=adaptive_sampler,
      current_state=current_state,
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      trace_fn=lambda current_state, kernel_results: kernel_results)

t0 = time.time()
samples, kernel_results = do_sampling()
t1 = time.time()
print("Inference ran in {:.2f}s.".format(t1-t0))

#print(samples)
#print(kernel_results.shape)

)"

py_run_string(bigstring)
# this create output as "samples"
# Clean up global environment
rm(data_l, envir = .GlobalEnv)

#extract out parameter phi from mcmc output
if(1){
  parid<-5
  par_samples<-py$samples[[parid]]; # samples indexes 1-no. params. this is 5th parameter all chains
  n_samples<-dim(par_samples)[1] # PER CHAIN
  chain1<-rep(NA,n_samples);
  chain2<-rep(NA,n_samples);
  chain3<-rep(NA,n_samples);
  chain4<-rep(NA,n_samples);

  for(i in 0:(n_samples-1)){ # 0-indexing
    chain1[i+1]<-par_samples[[i]]$numpy()[1]
    chain2[i+1]<-par_samples[[i]]$numpy()[2]
    chain3[i+1]<-par_samples[[i]]$numpy()[3]
    chain4[i+1]<-par_samples[[i]]$numpy()[4]
  }

  par(mfrow=c(1,2))
  plot(chain1,type="l",col="black")
  lines(chain2,col="red")
  lines(chain3,col="magenta")
  lines(chain4,col="blue")

  plot(density(chain1),col="black")
  lines(density(chain2),col="red")
  lines(density(chain3),col="magenta")
  lines(density(chain4),col="blue")
}



}

library(reticulate)
library(rstanarm)
library(tfprobability)

glm_negbin(thedata=roaches)
