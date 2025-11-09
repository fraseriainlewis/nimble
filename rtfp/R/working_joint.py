import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
tfb = tfp.bijectors
import numpy as np
import pandas as pd
import time

roaches=pd.read_csv('roaches.csv')

roach_data=tf.convert_to_tensor(roaches.roach1/100, dtype = tf.float32)
trt_data=tf.convert_to_tensor(roaches.treatment, dtype = tf.float32)
snr_data=tf.convert_to_tensor(roaches.senior, dtype = tf.float32)
exposure_data=tf.convert_to_tensor(roaches.exposure2, dtype = tf.float32)
y_data=tf.convert_to_tensor(roaches.y, dtype = tf.float32)

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
                 initial_vertex=start, func_tolerance=1e-08)
                 
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
num_burnin_steps=500
 
sampler = tfp.mcmc.TransformedTransitionKernel(
    tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=log_prob_fn,
        step_size=tf.cast(0.5, tf.float32)), #tf.cast(0.1, tf.float64)),
    bijector=unconstraining_bijectors
    )

adaptive_sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=sampler,
    num_adaptation_steps=int(0.8 * num_burnin_steps),
    target_accept_prob=tf.cast(0.75, tf.float64))


initial_state = optim_results.position
print(initial_state)

current_state=[tf.constant([2.8],dtype = tf.float32),
               tf.constant([1.3],dtype = tf.float32),
               tf.constant([-0.76],dtype = tf.float32),
               tf.constant([-0.33],dtype = tf.float32),
               tf.constant([0.27],dtype = tf.float32)]#,
#                     tf.constant(1.0,dtype = tf.float32),
#                     tf.constant(1.0,dtype = tf.float32),
#                     tf.constant(1.0,dtype = tf.float32),
#                     tf.constant(1.0,dtype = tf.float32)]
print(current_state)

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



