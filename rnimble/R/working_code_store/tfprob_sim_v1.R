library(tensorflow)
library(tfprobability)
library(zeallot)
library(purrr)
library(emdbook)
reticulate::py_require("tensorflow-metal")
data("ReedfrogPred")
d <- ReedfrogPred
str(d)

n_tadpole_tanks <- nrow(d)
n_surviving <- d$surv
n_start <- d$density

model <- tfd_joint_distribution_sequential(
  list(
    # a_bar, the prior for the mean of the normal distribution of per-tank logits
    tfd_normal(loc = 0, scale = 1.5),
    # sigma, the prior for the variance of the normal distribution of per-tank logits
    tfd_exponential(rate = 1),
    # normal distribution of per-tank logits
    # parameters sigma and a_bar refer to the outputs of the above two distributions
    function(sigma, a_bar)
      tfd_sample_distribution(
        tfd_normal(loc = a_bar, scale = sigma),
        sample_shape = list(n_tadpole_tanks)
      ),
    # binomial distribution of survival counts
    # parameter l refers to the output of the normal distribution immediately above
    function(l)
      tfd_independent(
        tfd_binomial(total_count = n_start, logits = l),
        reinterpreted_batch_ndims = 1
      )
  )
)

s <- model %>% tfd_sample(2)
s

model %>% tfd_log_prob(s)

logprob <- function(a, s, l)
  model %>% tfd_log_prob(list(a, s, l, n_surviving))

# number of steps after burnin
n_steps <- 500
# number of chains
n_chain <- 4
# number of burnin steps
n_burnin <- 500

hmc <- mcmc_hamiltonian_monte_carlo(
  target_log_prob_fn = logprob,
  num_leapfrog_steps = 3,
  # one step size for each parameter
  step_size = list(0.1, 0.1, 0.1),
) %>%
  mcmc_simple_step_size_adaptation(target_accept_prob = 0.8,
                                   num_adaptation_steps = n_burnin)

# initial values to start the sampler
c(initial_a, initial_s, initial_logits, .) %<-% (model %>% tfd_sample(n_chain))

# optionally retrieve metadata such as acceptance ratio and step size
trace_fn <- function(state, pkr) {
  list(pkr$inner_results$is_accepted,
       pkr$inner_results$accepted_results$step_size)
}

run_mcmc <- function(kernel) {
  kernel %>% mcmc_sample_chain(
    num_results = n_steps,
    num_burnin_steps = n_burnin,
    current_state = list(initial_a, tf$ones_like(initial_s), initial_logits),
    trace_fn = trace_fn
  )
}

run_mcmc <- tf_function(run_mcmc)
res <- run_mcmc(hmc)

###############################


tensorflow::tf_config()
tensorflow::tf_version()

library(rstanarm)
data(mtcars)
# Input data: 10 data points
x1_data <- tf$constant(mtcars$wt, dtype = tf$float32)
x2_data <- tf$constant(mtcars$am, dtype = tf$float32)
y_data <- tf$constant(mtcars$mpg, dtype = tf$float32)
n_obs <- nrow(mtcars)

# Define the joint distribution
m <- tfd_joint_distribution_sequential(
  list(
    # Intercept (alpha)
    tfd_normal(loc = 0, scale = 5),

    # Slope (beta_wt)
    tfd_normal(loc = 0, scale = 1),
    # Slope (beta_am)
    tfd_normal(loc = 0, scale = 1),

    # Noise standard deviation (sigma)
    tfd_exponential(rate = 1),

    # Observations: y = alpha + beta * x + noise
    function(sigma, beta_am,beta_wt, alpha) {
      # Compute linear mean: mu = alpha + beta * x
      # When sampling 3 times:
      # alpha: (3,), beta: (3,), x_data: (10,)
      # Need to broadcast to (3, 10)

      alpha_expanded <- tf$expand_dims(alpha, -1L)  # (3, 1)
      beta_wt_expanded <- tf$expand_dims(beta_wt, -1L)    # (3, 1)
      beta_am_expanded <- tf$expand_dims(beta_am, -1L)    # (3, 1)

      mu <- alpha_expanded + beta_wt_expanded * x1_data + + beta_am_expanded * x2_data  # (3, 1) + (3, 1) * (10,) = (3, 10)

      # Expand sigma to broadcast
      sigma_expanded <- tf$expand_dims(sigma, -1L)  # (3, 1)

      # Create distribution for observations
      tfd_independent(
        tfd_normal(loc = mu, scale = sigma_expanded),
        reinterpreted_batch_ndims = 1L
      )
    }
  )
)

# Simulate 3 samples
m %>% tfd_sample(3L)

s<-m %>% tfd_sample(2)


m %>% tfd_log_prob(s)

logprob <- function(alpha, beta_wt,beta_am,phi)
  m %>% tfd_log_prob(list(alpha, beta_wt, beta_am,phi, y_data))

logprob(0.1,0.2,0.3,0.5)



# number of steps after burnin
n_steps <- 100000
# number of chains
n_chain <- 1
# number of burnin steps
n_burnin <- 10000

if(TRUE){hmc <- mcmc_hamiltonian_monte_carlo(
  target_log_prob_fn = logprob,
  num_leapfrog_steps = 3,
  # one step size for each parameter
  step_size = list(0.5, 0.5, 0.5,0.5),
  seed=99999
) %>%
  mcmc_simple_step_size_adaptation(target_accept_prob = 0.8,
                                   num_adaptation_steps = n_burnin)
}
if(FALSE){hmc_kernel <- mcmc_hamiltonian_monte_carlo(
  target_log_prob_fn = logprob,
  num_leapfrog_steps = 3,
  # one step size for each parameter
  step_size = list(0.5, 0.5, 0.5,0.5),
  seed=99999
)

adaptive_hmc_kernel <- tfp$mcmc$DualAveragingStepSizeAdaptation(
  inner_kernel = hmc_kernel,
  num_adaptation_steps = as.integer(n_burnin),
  target_accept_prob = 0.8
)
}

# initial values to start the sampler
c(alpha, beta_wt,beta_am,phi, .) %<-% (m %>% tfd_sample(n_chain))


# optionally retrieve metadata such as acceptance ratio and step size
trace_fn <- function(state, pkr) {
  list(pkr$inner_results$is_accepted,
       pkr$inner_results$accepted_results$step_size)
}

run_mcmc <- function(kernel) {
  kernel %>% mcmc_sample_chain(
    num_results = n_steps,
    num_burnin_steps = n_burnin,
    current_state = list(alpha, beta_wt,beta_am,phi),
    trace_fn = trace_fn
  )
}

run_mcmc <- tf_function(run_mcmc)
res <- run_mcmc(hmc)

mcmc_trace <- res$all_states

par(mfrow=c(2,2))
plot(mcmc_trace[[1]],type="l")
plot(mcmc_trace[[2]],type="l")
plot(mcmc_trace[[3]],type="l")
plot(mcmc_trace[[4]],type="l")

rmod<-lm(mpg~1+wt+am,data=mtcars)
summary(rmod)

mean(mcmc_trace[[1]])
mean(mcmc_trace[[2]])
mean(mcmc_trace[[3]])
mean(mcmc_trace[[4]])


##########################################
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# --- 1. Simulate data for the Negative Binomial Regression Model ---
# (Since no data is provided, we simulate a dataset that matches the description)

set.seed(12345)

# --- Define the Stan model as a string in R ---

stan_model_string <- "
data {
  int<lower=1> N;                 // Number of observations
  int<lower=1> M;                 //number of predictors excl intercept
  array[N] real<lower=0> y;         // Response variable (counts)
  vector[N] wt;           // Continuous predictor
  vector[N] am;    // Binary predictor
}


parameters {
  real alpha;                     // Intercept
  real beta_wt;               // Coefficient for roach1
  real beta_am;            // Coefficient for treatment
  real<lower=0> phi;              // Negative Binomial overdispersion parameter
}

transformed parameters {
  array[N] real mu;           // Log of the mean parameter
  for (i in 1:N) {
    // linear model
    mu[i] = alpha +
                beta_wt * wt[i] +
                beta_am * am[i] ;
  }
}

model {
  // --- Priors ---
  // Weakly informative priors for coefficients and intercept
  alpha ~ normal(0, 5.0);           // Prior for intercept
  beta_wt ~ normal(0,1);     // Prior for roach1 coefficient
  beta_am ~ normal(0, 1);  // Prior for treatment coefficient
  phi ~ exponential(1); // this is actually 1/ rescaled scale

  // --- Likelihood ---
  // Negative Binomial likelihood, using the log-link function for the mean
  y ~  normal(mu, phi);
}

"

# --- 3. Prepare data for Stan ---
# The data needs to be provided as a list for rstan::stan()
stan_data <- list(
  N = nrow(mtcars),
  M = 3, # number of passed hyperpriors
  #rescaled_sd=c(beta_prior_scale,1/sd_prior_scale),# 1/ as prior uses rate
  y = mtcars$mpg,
  wt = mtcars$wt,
  am = mtcars$am
)

# --- 4. Fit the Stan model ---
# Use rstan::stan() to compile and sample from the model
fit <- stan(
  model_code = stan_model_string,
  data = stan_data,
  chains = 4,         # Number of MCMC chains
  warmup = 1000,      # Number of warmup iterations per chain
  iter = 10000,        # Total iterations per chain (warmup + sampling)
  thin = 1,           # Thinning rate
  seed = 12345,          # For reproducibility
  control = list(adapt_delta = 0.95, max_treedepth = 15) # Adjust for sampling issues if needed
)

# --- 5. Extract main parameters and produce density plots ---
res2<-extract(fit,par=c("alpha","beta_wt"," beta_am","phi"))

mean(res2$alpha)
mean(res2$beta_wt)
mean(res2$beta_am)
mean(res2$phi)

mean(mcmc_trace[[1]])
mean(mcmc_trace[[2]])
mean(mcmc_trace[[3]])
mean(mcmc_trace[[4]])



