# This file compares the use of linalg approach with design matrix in model
# rather than do manually. There are a few quirks to be aware of
# in particular the dimension matching is not obvious in how R converts
# to python - some trial and error. also the output from the functions
# can change from shape=(1) to shape=() which impacts the optimizers
#
#
#rm(list=ls())
## Tensorflow - bonus
library(tensorflow)
library(tfprobability)

library(rstanarm)
data(roaches)
roaches$roach1 <- roaches$roach1 / 100
rescaled_sd<-c(2.5,2.5,2.5)

# APPROACH 1. use vectors and build up logmu=.... using matrix*scalar mult
# this used model - prob dist - m below
# Input data
roach_data <- tf$constant(roaches$roach1, dtype = tf$float32)
trt_data <- tf$constant(roaches$treatment, dtype = tf$float32)
snr_data <- tf$constant(roaches$senior, dtype = tf$float32)
exposure_data <- tf$constant(roaches$exposure2, dtype = tf$float32)
y_data<-tf$constant(roaches$y, dtype = tf$float32)

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
