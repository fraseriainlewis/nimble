# Hello, world!
#
# This is an example function named 'hello'
# which prints 'Hello, world!'.
#
# You can learn more about package authoring with RStudio at:
#
#   https://r-pkgs.org
#
# Some useful keyboard shortcuts for package authoring:
#
#   Install Package:           'Cmd + Shift + B'
#   Check Package:             'Cmd + Shift + E'
#   Test Package:              'Cmd + Shift + T'

.onLoad <- function(libname, pkgname) {
  reticulate::py_require("tensorflow")
  reticulate::py_require("tf_keras")
  reticulate::py_require("tensorflow_probability")
}

#' Add together two numbers
#'
#' @returns The sum of `x` and `y`
#' @export
#' @examples
#'
#' test_call_to_tfd()
test_call_to_tfd <- function() {
  print("calling....\n")
  return(tfd_bernoulli(probs=0.5)%>%tfd_sample(5L))
}

#' Add together two numbers
#'
#' @returns The sum of `x` and `y`
#' @export
#' @examples
#'
#' test_call_to_tfd()
script_tfd <- function(x) {
  print("calling....\n")
  a<-10.0
  loc_x<-x;
  assign("loc_x", loc_x, envir = .GlobalEnv)
  #print(tfd_bernoulli(probs=0.5)%>%tfd_sample(5L))
  mystring<-"
import tensorflow as tf
import tensorflow_probability as tfd

fromr= r.a + 10
fromr2=r.roaches
print(type(r.roaches))
print(r.loc_x)
#print()
"
  py_run_string(mystring)
cat("got=",py$fromr,"\n")

# Clean up global environment
rm(loc_x, envir = .GlobalEnv)

}

#' Add together two numbers
#'
#' @description A short description...
#' @param thedata data.frame of the data to fit to the model
#' @returns currently nothing
#' @examples
#' library(rstanarm)
#' data(roaches)
#' roaches$roach1<-roaches$roach1/100;# manual
#' glm_negbin(thedata=roaches)

#' @export
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

#' Add together two numbers
#'
#' @description A short description...
#' @param data data.frame of the data to fit to the model
#' @param prior vector of prior sd
#' @returns currently nothing
#' @export
#' @examples
#' \dontrun{
#' # Basic usage
#' library(rstanarm)
#' data(roaches)
#' rtfp_negbin(data=roaches, prior=c(2.5,2.5,2.5))
#' }
#'
rtfp_negbin<-function(data=NULL,prior=NULL) {

# APPROACH 1. use vectors and build up logmu=.... using matrix*scalar mult
# this used model - prob dist - m below
# Input data
roaches<-data
roach_data <- tf$constant(roaches$roach1, dtype = tf$float32)
trt_data <- tf$constant(roaches$treatment, dtype = tf$float32)
snr_data <- tf$constant(roaches$senior, dtype = tf$float32)
exposure_data <- tf$constant(roaches$exposure2, dtype = tf$float32)
y_data<-tf$constant(roaches$y, dtype = tf$float32)

rescaled_sd<-prior;

# APPROACH 1. Define the joint distribution without matrix mult
m1 <- tfd_joint_distribution_sequential(
  list(
    # Intercept (alpha)
    tfd_normal(loc = 0, scale = 5),
    # Slope (beta_roach1)
    tfd_normal(loc = 0, scale = rescaled_sd[1]),
    # Slope (beta_treatment)
    tfd_normal(loc = 0, scale = rescaled_sd[2]),
    # Slope (beta_senior)
    tfd_normal(loc = 0, scale = rescaled_sd[3]),
    # Noise standard deviation (phi)
    tfd_exponential(rate = 1),

    # Observations: y = alpha + beta * x + noise
    function(phi, beta_senior,beta_treatment, beta_roach1,alpha) {
      # Compute linear mean: mu = alpha + beta * x
      # When sampling 3 times:
      # alpha: (3,), beta: (3,), x_data: (10,)
      # Need to broadcast to (3, 10)

      alpha_expanded <- tf$expand_dims(alpha, -1L)  # (3, 1)
      beta_roach1_expanded <- tf$expand_dims(beta_roach1, -1L)    # (3, 1)
      beta_treatment_expanded <- tf$expand_dims(beta_treatment, -1L)    # (3, 1)
      beta_senior_expanded <- tf$expand_dims(beta_senior, -1L)    # (3, 1)
      beta_expos_expanded <- tf$expand_dims(1.0, -1L)

      logmu <- alpha_expanded + beta_roach1_expanded * roach_data +
        beta_treatment_expanded * trt_data +
        beta_senior_expanded * snr_data +log(exposure_data)

      # (3, 1) + (3, 1) * (10,) = (3, 10)

      # Expand sigma to broadcast
      phi_expanded <- tf$expand_dims(phi, -1L)  # (3, 1)

      mu = exp(logmu)
      #r = tf$expand_dims(1.0, -1L)/ phi_expanded;  # total_count = 5
      #probs <- r / (r + mu)

      prob <- phi_expanded/(phi_expanded+mu)
      #prob<-(phi_expanded)/(mu+phi_expanded)
      # Create distribution for observations
      tfd_independent(
        #tfd_normal(loc = mu, scale = sigma_expanded),
        #mu = exp(mu)
        #phi <- 0.2  # scale/overdispersion

        #r = tf$expand_dims(1.0, -1L)/ sigma_expanded;  # total_count = 5
        #probs <- r / (r + mu)
        tfd_negative_binomial(total_count = phi_expanded, probs = 1-prob),
        reinterpreted_batch_ndims = 1L
      )
    }
  )
)

# Approach 1.
tf$random$set_seed(9999)
s1<-m1 %>% tfd_sample(1L)
print(s1[[1]])
tf$random$set_seed(9999)
s1<-m1 %>% tfd_sample(1L)
print(s1[[1]])

# Approach 1.
logprob <- function(alpha, beta_roach1,beta_treatment,beta_senior,phi)
  m1 %>% tfd_log_prob(list(alpha, beta_roach1,beta_treatment,beta_senior,phi,y_data))

logprob(0.1,0.2,0.3,0.5,0.1)


# Approach 1.
neg_logprob <- tf_function(function(mypar){
  alpha<-mypar[1]; beta_roach1<-mypar[2];beta_treatment<-mypar[3];beta_senior<-mypar[4];phi<-mypar[5];
  x<- -tfd_log_prob(m1,list(alpha, beta_roach1,beta_treatment,beta_senior,phi,y_data))
  return(x)})
neg_logprob(c(0.1,0.2,0.3,0.5,0.1))

#### get starting values by find MLE
start = tf$constant(c(0.1,0.2,0.3,0.5,0.1))  # Starting point for the search.
optim_results = tfp$optimizer$nelder_mead_minimize(
  neg_logprob, initial_vertex=start, func_tolerance=1e-08,
  batch_evaluate_objective=FALSE)#,max_iterations=5000)
optim_results$initial_objective_values
optim_results$objective_value
optim_results$position

print(optim_results)


# number of steps after burnin
n_steps <- 20000
# number of chains
n_chain <- 10
# number of burnin steps
n_burnin <- 10000

# bijector to map constrained parameters to real
unconstraining_bijectors <- list(
  tfb_identity(), tfb_identity(), tfb_identity(),tfb_identity(),tfb_exp())

nuts <- mcmc_no_u_turn_sampler(
  target_log_prob_fn = logprob,
  step_size = list(0.5, 0.5, 0.5,0.5,0.5)
) %>%
  mcmc_transformed_transition_kernel(bijector = unconstraining_bijectors) %>%
  mcmc_dual_averaging_step_size_adaptation(
    num_adaptation_steps = round(n_burnin*0.8),
    step_size_setter_fn = function(pkr, new_step_size)
      pkr$`_replace`(
        inner_results = pkr$inner_results$`_replace`(step_size = new_step_size)),
    step_size_getter_fn = function(pkr) pkr$inner_results$step_size,
    log_accept_prob_getter_fn = function(pkr) pkr$inner_results$log_accept_ratio
  )

hmc <- mcmc_hamiltonian_monte_carlo(
  target_log_prob_fn = logprob,
  num_leapfrog_steps = 3,
  # one step size for each parameter
  step_size = list(0.5, 0.5, 0.5,0.5,0.5),
  seed=99999
) %>% mcmc_dual_averaging_step_size_adaptation(
  num_adaptation_steps = round(n_burnin*0.8),
  target_accept_prob = 0.75,
  exploration_shrinkage = 0.05,
  step_count_smoothing = 10,
  decay_rate = 0.75
)

#c(alpha, beta_roach1,beta_treatment,beta_senior,phi, .) %<-% optim_results$position
res<-matrix(rep(optim_results$position,n_chain),nrow=n_chain,byrow=TRUE)
mylist<-apply(res,2,FUN=function(a){return(tf$constant(array(a),dtype=tf$float32))})


run_mcmc <- tf_function(function(kernel) {
  kernel %>% mcmc_sample_chain(
    num_results = n_steps,
    num_burnin_steps = n_burnin,
    current_state = list(mylist[[1]], mylist[[2]],mylist[[3]],mylist[[4]],mylist[[5]]),
    seed=9999#,
    #parallel_iterations=1
  )
}
)
set.seed(9999)
#run_mcmc <- tf_function(run_mcmc)
print(system.time(mcmc_trace <- run_mcmc(hmc)))
mcmc_trace_c<-lapply(mcmc_trace,FUN=function(a){return(c(as.matrix(a)))})

alpha<-mcmc_trace_c[[1]]
beta_roach1<-mcmc_trace_c[[2]]
beta_treatment<-mcmc_trace_c[[3]]
beta_senior<-mcmc_trace_c[[4]]
phi<-mcmc_trace_c[[5]]
plot(density(alpha))

cat("Approach 2 - mat mul\n")

# APPROACH 2. use design matrix with intercept and offset included in this
# Input data
roach_data <- tf$constant(roaches$roach1, dtype = tf$float32)
trt_data <- tf$constant(roaches$treatment, dtype = tf$float32)
snr_data <- tf$constant(roaches$senior, dtype = tf$float32)
exposure_data <- tf$constant(roaches$exposure2, dtype = tf$float32)
y_data<-tf$constant(roaches$y, dtype = tf$float32)

X_mat<-cbind(1, # include intercept
             as.matrix(roaches[,c("roach1","treatment","senior",
                                  "exposure2")])) # trailing additive term
X_mat[,5]<-log(X_mat[,5]); # as logmu = ....+log(exposure2)
X_mat<-tf$constant(X_mat, dtype = tf$float32); # explicit conversion needed to 32-bit

# APPROACH 2. Define the joint distribution with matrix mult

m2 <- tfd_joint_distribution_sequential(
  list(
    # Intercept (alpha)
    tfd_normal(loc = 0, scale = 5),
    # Slope (beta_roach1)
    tfd_normal(loc = 0, scale = rescaled_sd[1]),
    # Slope (beta_treatment)
    tfd_normal(loc = 0, scale = rescaled_sd[2]),
    # Slope (beta_senior)
    tfd_normal(loc = 0, scale = rescaled_sd[3]),
    # Noise standard deviation (phi)
    tfd_exponential(rate = 1),

    # Observations: y = alpha + beta * x + noise
    function(phi, beta_senior,beta_treatment, beta_roach1,alpha) {
      # Compute linear mean: mu = alpha + beta * x
      # When sampling 3 times:
      # alpha: (3,), beta: (3,), x_data: (10,)
      # Need to broadcast to (3, 10)

      # note: cbind is needed NOT c()
      # and 1.0 scalar not vector - appears broadcasting does rest
      logmu<-tf$linalg$matvec(X_mat,
                              cbind(alpha,beta_roach1,beta_treatment,beta_senior,1.0))

      # (3, 1) + (3, 1) * (10,) = (3, 10)

      # Expand sigma to broadcast
      phi_expanded <- tf$expand_dims(phi, -1L)  # (3, 1)

      mu = exp(logmu)
      #r = tf$expand_dims(1.0, -1L)/ phi_expanded;  # total_count = 5
      #probs <- r / (r + mu)

      prob <- phi_expanded/(phi_expanded+mu)
      #prob<-(phi_expanded)/(mu+phi_expanded)
      # Create distribution for observations
      tfd_independent(
        #tfd_normal(loc = mu, scale = sigma_expanded),
        #mu = exp(mu)
        #phi <- 0.2  # scale/overdispersion

        #r = tf$expand_dims(1.0, -1L)/ sigma_expanded;  # total_count = 5
        #probs <- r / (r + mu)
        tfd_negative_binomial(total_count = phi_expanded, probs = 1-prob),
        reinterpreted_batch_ndims = 1L
      )
    }
  )
)

# Approach 2.
tf$random$set_seed(9999)
s1<-m2 %>% tfd_sample(1L)
print(s1[[1]])
tf$random$set_seed(99099)
s2<-m2 %>% tfd_sample(1L)
print(s2[[1]])
# these give same form

# Approach 2.
logprob <- function(alpha, beta_roach1,beta_treatment,beta_senior,phi)
  m2 %>% tfd_log_prob(list(alpha, beta_roach1,beta_treatment,beta_senior,phi,y_data))

logprob(0.1,0.2,0.3,0.5,0.1)

## IMPORTANT NOTE: In approach 1 the result is a SCALAR, in 2 it's a VECTOR

# Approach 2. NEEDS tf$squeeze, otherwise tfp$optimizer fails B
# the MCMC estimate seems happy with either approach
neg_logprob <- tf_function(function(mypar){
  alpha<-mypar[1]; beta_roach1<-mypar[2];beta_treatment<-mypar[3];beta_senior<-mypar[4];phi<-mypar[5];
  x<- -tf$squeeze(tfd_log_prob(m2,list(alpha, beta_roach1,beta_treatment,beta_senior,phi,y_data)))
  return(x)})

neg_logprob(c(0.1,0.2,0.3,0.5,0.1))

#### get starting values by find MLE
start = tf$constant(c(0.1,0.2,0.3,0.5,0.1))  # Starting point for the search.
optim_results = tfp$optimizer$nelder_mead_minimize(
  neg_logprob, initial_vertex=start, func_tolerance=1e-08,
  batch_evaluate_objective=FALSE)#,max_iterations=5000)
optim_results$initial_objective_values
optim_results$objective_value
optim_results$position

# number of steps after burnin
n_steps <- 20000
# number of chains
n_chain <- 10
# number of burnin steps
n_burnin <- 10000


# bijector to map constrained parameters to real
unconstraining_bijectors <- list(
  tfb_identity(), tfb_identity(), tfb_identity(),tfb_identity(),tfb_exp())

nuts <- mcmc_no_u_turn_sampler(
  target_log_prob_fn = logprob,
  step_size = list(0.5, 0.5, 0.5,0.5,0.5)
) %>%
  mcmc_transformed_transition_kernel(bijector = unconstraining_bijectors) %>%
  mcmc_dual_averaging_step_size_adaptation(
    num_adaptation_steps = round(n_burnin*0.8),
    step_size_setter_fn = function(pkr, new_step_size)
      pkr$`_replace`(
        inner_results = pkr$inner_results$`_replace`(step_size = new_step_size)),
    step_size_getter_fn = function(pkr) pkr$inner_results$step_size,
    log_accept_prob_getter_fn = function(pkr) pkr$inner_results$log_accept_ratio
  )

hmc <- mcmc_hamiltonian_monte_carlo(
  target_log_prob_fn = logprob,
  num_leapfrog_steps = 3,
  # one step size for each parameter
  step_size = list(0.5, 0.5, 0.5,0.5,0.5),
  seed=99999
) %>% mcmc_dual_averaging_step_size_adaptation(
  num_adaptation_steps = round(n_burnin*0.8),
  target_accept_prob = 0.75,
  exploration_shrinkage = 0.05,
  step_count_smoothing = 10,
  decay_rate = 0.75
)


#c(alpha, beta_roach1,beta_treatment,beta_senior,phi, .) %<-% optim_results$position
res<-matrix(rep(optim_results$position,n_chain),nrow=n_chain,byrow=TRUE)
mylist<-apply(res,2,FUN=function(a){return(tf$constant(array(a),dtype=tf$float32))})

run_mcmc <- tf_function(function(kernel) {
  kernel %>% mcmc_sample_chain(
    num_results = n_steps,
    num_burnin_steps = n_burnin,
    current_state = list(mylist[[1]], mylist[[2]],mylist[[3]],mylist[[4]],mylist[[5]]),
    seed=100#,
    #parallel_iterations=1
  )
}
)

#run_mcmc <- tf_function(run_mcmc)
print(system.time(mcmc_trace <- run_mcmc(hmc)))
mcmc_trace_c<-lapply(mcmc_trace,FUN=function(a){return(c(as.matrix(a)))})

alpha<-mcmc_trace_c[[1]]
beta_roach1<-mcmc_trace_c[[2]]
beta_treatment<-mcmc_trace_c[[3]]
beta_senior<-mcmc_trace_c[[4]]
phi<-mcmc_trace_c[[5]]

lines(density(alpha),col="magenta")

}

