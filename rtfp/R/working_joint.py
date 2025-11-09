import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import numpy as np
import pandas as pd

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
#tf.random.set_seed(9999)
#a=model.sample()
print(model)
a=model.sample()
print(a)

a=model.sample()
print(a)

a=model.sample()
print(a)

def target_log_prob_fn(alpha, beta_roach,beta_treatment,beta_senior,phi):
  """Unnormalized target density as a function of states."""
  return model.log_prob((
      alpha, beta_roach,beta_treatment,beta_senior,phi, y_data))
         
print(target_log_prob_fn(0.1,0.2,0.3,0.5,0.1))
   
 
