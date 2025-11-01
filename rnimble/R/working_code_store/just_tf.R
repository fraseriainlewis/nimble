## Tensorflow - bonus
library(tensorflow)
library(tfprobability)

library(rstanarm)
data(roaches)
roaches$roach1 <- roaches$roach1 / 100
rescaled_sd<-c(2.5,2.5,2.5)

# Input data and centre this here before passing to tf
roach_data <- tf$constant(roaches$roach1, dtype = tf$float32)
trt_data <- tf$constant(roaches$treatment, dtype = tf$float32)
snr_data <- tf$constant(roaches$senior, dtype = tf$float32)
exposure_data <- tf$constant(roaches$exposure2, dtype = tf$float32)
y_data<-tf$constant(roaches$y, dtype = tf$float32)

X_mat<-cbind(1,as.matrix(roaches[,c("roach1","treatment","senior","exposure2")]))
X_mat[,5]<-log(X_mat[,5])
X_mat<-tf$constant(X_mat, dtype = tf$float32)

#// Log-linear model with log(exposure2) as an offset
#log_mu[i] = alpha +
#  beta_roach1 * roach1[i] +
#  beta_treatment * treatment[i] +
#  beta_senior * senior[i] +
#  log(exposure2[i]); // Offset


# Define the joint distribution
m <- tfd_joint_distribution_sequential(
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

      logmu<-tf$linalg$matvec(X_mat,cbind(alpha,beta_roach1,beta_treatment,beta_senior,1.0))


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

m %>% tfd_sample(1L)
s<-m %>% tfd_sample(1L)

# Define the joint distribution
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


# Simulate 3 samples
#s<-m %>% tfd_sample(1L)
# s<-m %>% tfd_sample(2)
# m %>% tfd_log_prob(s)

#table(as.numeric(s[[6]]))

#intercept_0=alpha + beta_wt*-mean_wt + beta_am*-mean_am;

logprob <- function(alpha, beta_roach1,beta_treatment,beta_senior,phi)
  m %>% tfd_log_prob(list(alpha, beta_roach1,beta_treatment,beta_senior,phi,y_data))

logprob(0.1,0.2,0.3,0.5,0.1)


neg_logprob <- tf_function(function(mypar){
  alpha<-mypar[1]; beta_roach1<-mypar[2];beta_treatment<-mypar[3];beta_senior<-mypar[4];phi<-mypar[5];
  x<- -tf$squeeze(tfd_log_prob(m,list(alpha, beta_roach1,beta_treatment,beta_senior,phi,y_data)))
  #x<- -tfd_log_prob(m,list(alpha, beta_roach1,beta_treatment,beta_senior,phi,y_data))
  return(x)})

neg_logprob(c(0.1,0.2,0.3,0.5,0.1))

library(reticulate)
start = tf$constant(c(0.1,0.2,0.3,0.5,0.1))  # Starting point for the search.
optim_results = tfp$optimizer$nelder_mead_minimize(
  neg_logprob, initial_vertex=start, func_tolerance=1e-08,
  batch_evaluate_objective=FALSE)#,max_iterations=5000)
optim_results$initial_objective_values
optim_results$objective_value
optim_results$position

#res<-optim(c(0.1,0.2,0.3,0.5,0.1),fn=neglogprob,method="Nelder-Mead")


# number of steps after burnin
n_steps <- 20000
# number of chains
n_chain <- 4
# number of burnin steps
n_burnin <- 10000

set.seed(99999)

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
  decay_rate = 0.75,
  step_size_setter_fn = NULL,
  step_size_getter_fn = NULL,
  log_accept_prob_getter_fn = NULL,
  validate_args = FALSE,
  name = NULL#,
)

# initial values to start the sampler - from prior
#c(alpha, beta_roach1,beta_treatment,beta_senior,phi, .) %<-% (m %>% tfd_sample(n_chain))

#c(alpha, beta_roach1,beta_treatment,beta_senior,phi, .) %<-% optim_results$position
res<-matrix(rep(optim_results$position,n_chain),nrow=n_chain,byrow=TRUE)
mylist<-apply(res,2,as_tensor,dtype=tf$float32)


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
system.time(mcmc_trace <- run_mcmc(hmc))
mcmc_trace_c<-lapply(mcmc_trace,FUN=function(a){return(c(as.matrix(a)))})

alpha<-mcmc_trace_c[[1]]
beta_roach1<-mcmc_trace_c[[2]]
beta_treatment<-mcmc_trace_c[[3]]
beta_senior<-mcmc_trace_c[[4]]
phi<-mcmc_trace_c[[5]]
plot(density(alpha))
lines(density(alpha,col="red"))
