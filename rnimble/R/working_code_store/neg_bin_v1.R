rm(list=ls())
setwd("/Users/work/rstan_nimble_proj")
### rstan nimble package project
library(rstan)
library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default())
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

library(rstanarm)
data(roaches)

# Rescale
# note for models that do internal predictor centering then need location shift back for intecept
# i.e y = a + b*(x1-mean) + c*(x2-mean)
# so want a at x1=0 and x2=0 so E(y) = mean(a) + mean(b)*-mean(x1) + mean(c)*-mean(x2) etc.
# this centering does not appear to be used in neg bin - as y is counts seem reasonable
# observe lambda*t = a + b + c but want just lambda, i.e. Y=lambda*t so lambda = Y/t
# log(lambda) = a + b + c  + log(exposure)
# P(X=x) = lambda^x exp(-lambda)/x!  lambda = lambda2*t
# 

roaches$roach1 <- roaches$roach1 / 100
# Estimate original model
glm1 <- glm(y ~ roach1 + treatment + senior, offset = log(exposure2), 
            data = roaches, family = poisson)
# Estimate Bayesian version with stan_glm
stan_glm1 <- stan_glm(y ~ roach1 + treatment + senior, offset = log(exposure2),
                      data = roaches, family = neg_binomial_2, 
                      prior = normal(0, 2.5), 
                      prior_intercept = normal(0, 5),
                      seed = 12345,
                      warmup = 10000,      # Number of warmup iterations per chain
                      iter = 100000,        # Total iterations per chain (warmup + sampling)
                      thin = 1,
                      chains = 4)           # Thinning rate)
res_m<-as.matrix(stan_glm1)
summary(res_m[,"(Intercept)"])
summary(res_m[,"roach1"])

## if priors are given only the beta coefs priors are autoscaled
## if priors given for beta then no autoscaling of any parameter is done
## centering is also not done for neg bin response
prior_scales<-prior_summary(stan_glm1)
# get the predictor adjusted scale - for nb dist no other rescaling is done
beta_prior_scale<-prior_scales$prior$scale
## THE ABOVE NEEDS FIXED if no PRIORS GIVEN
#beta_prior_scale<-prior_scales$prior$adjusted_scale


# log(l*n) = a + b, log(l)+log(n) = 
 
################################################################################
## use base rstan
## the below is from gemini
# Load necessary libraries
library(rstan)
library(ggplot2)
library(bayesplot)
library(dplyr) # For data manipulation if needed, e.g., tibble

# Set Stan options for better performance and to avoid recompilation
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
  array[N] int<lower=0> y;         // Response variable (counts)
  array[N] real<lower=0> roach1;           // Continuous predictor
  array[N] int<lower=0,upper=1> treatment; // Binary predictor
  array[N] int<lower=0,upper=1> senior;    // Binary predictor
  array[N] real<lower=0> exposure2; // Count offset variable (must be > 0)
  //Hyperparameters
  array[M] real<lower=0>rescaled_sd;// standard dev for predictors

}

parameters {
  real alpha;                     // Intercept
  real beta_roach1;               // Coefficient for roach1
  real beta_treatment;            // Coefficient for treatment
  real beta_senior;               // Coefficient for senior
  real<lower=0> phi;              // Negative Binomial overdispersion parameter
}

transformed parameters {
  array[N] real log_mu;           // Log of the mean parameter
  for (i in 1:N) {
    // Log-linear model with log(exposure2) as an offset
    log_mu[i] = alpha +
                beta_roach1 * roach1[i] +
                beta_treatment * treatment[i] +
                beta_senior * senior[i] +
                log(exposure2[i]); // Offset
  }
}

model {
  // --- Priors ---
  // Weakly informative priors for coefficients and intercept
  alpha ~ normal(0, 5.0);           // Prior for intercept
  beta_roach1 ~ normal(0,rescaled_sd[1] );     // Prior for roach1 coefficient
  beta_treatment ~ normal(0, rescaled_sd[2]);  // Prior for treatment coefficient
  beta_senior ~ normal(0, rescaled_sd[3]);     // Prior for senior coefficient

  phi ~ exponential(1); 

  // --- Likelihood ---
  // Negative Binomial likelihood, using the log-link function for the mean
  y ~  neg_binomial_2_log(log_mu, phi);
}

generated quantities {
  // Can include posterior predictions or log-likelihood here if desired for model checking
}
"

# --- 3. Prepare data for Stan ---
# The data needs to be provided as a list for rstan::stan()
stan_data <- list(
  N = nrow(roaches),
  M = 3, # number of predictors
  rescaled_sd=beta_prior_scale,
  y = roaches$y,
  roach1 = roaches$roach1,
  treatment = roaches$treatment,
  senior = roaches$senior,
  exposure2 = roaches$exposure2
)

# --- 4. Fit the Stan model ---
# Use rstan::stan() to compile and sample from the model
fit <- stan(
  model_code = stan_model_string,
  data = stan_data,
  chains = 4,         # Number of MCMC chains
  warmup = 10000,      # Number of warmup iterations per chain
  iter = 100000,        # Total iterations per chain (warmup + sampling)
  thin = 1,           # Thinning rate
  seed = 12345,          # For reproducibility
  control = list(adapt_delta = 0.95, max_treedepth = 15) # Adjust for sampling issues if needed
)

# --- 5. Extract main parameters and produce density plots ---
res2<-extract(fit,par=c("alpha","beta_roach1"," beta_treatment","beta_senior","phi"))
print(summary(res2$alpha))
print(summary(res_m[,"(Intercept)"]))
print(summary(res2$beta_roach1))
print(summary(res_m[,"roach1"]))

#par(mfrow=c(1,2))
#plot(density(res_m[,"(Intercept)"]))
#lines(density(res2$alpha),col="red")

#plot(density(res_m[,"roach1"]))
#lines(density(res2$beta_roach1),col="blue")

################################################################################

# Load necessary libraries
library(nimble)
library(coda) # For MCMC diagnostics and plotting


# --- Define the Nimble Model Code ---
# This block defines the statistical model using Nimble's DSL.
# The negative binomial distribution in Nimble (dnegbin) is parameterized
# by `prob` and `size`. Its mean is `size * (1-prob) / prob`.
# We relate this mean to a linear predictor `log_lambda`.
neg_binom_regression_code <- nimbleCode({
  # Priors for regression coefficients and intercept (weak priors)
  # Using normal distributions with large standard deviations (small precision/tau)
  intercept ~ dnorm(0, sd = 5) # Weakly informative prior for intercept
  beta_roach1 ~ dnorm(0, sd = 2.5) # Weakly informative prior for roach1 coefficient
  beta_treatment ~ dnorm(0, sd = 2.5) # Weakly informative prior for treatment coefficient
  beta_senior ~ dnorm(0, sd = 2.5) # Weakly informative prior for senior coefficient
  
  # Prior for the Negative Binomial dispersion parameter (size)
  # 'size' must be positive. A Gamma distribution is a common choice for scale parameters.
  # dgamma(shape, rate) with small shape and rate implies a weak prior.
  #size ~ dgamma(0.01, 0.01) # Weakly informative prior for dispersion parameter
  size ~ dexp(1.0)
  
  # Likelihood for each observation
  for (i in 1:N) {
    # Linear predictor on the log scale (log_lambda_expected)
    # exposure2 is an offset, so log(exposure2[i]) is added directly
    # with a coefficient fixed at 1.
    log_lambda_expected[i] <- intercept +
      beta_roach1 * roach1[i] +
      beta_treatment * treatment[i] +
      beta_senior * senior[i] +
      log(exposure2[i]) # Offset term
    
    # Convert log_lambda_expected to lambda (expected mean count)
    lambda[i] <- exp(log_lambda_expected[i])
    
    # Convert lambda and size to 'prob' parameter for dnegbin
    # dnegbin(prob, size) has mean = size * (1 - prob) / prob
    # So, lambda[i] = size * (1 - prob[i]) / prob[i]
    # Rearranging for prob[i]: prob[i] = size / (size + lambda[i])
    # An equivalent form is prob[i] = 1 / (1 + lambda[i] / size)
    prob[i] <- 1 / (1 + lambda[i] / size)
    
    # Negative Binomial likelihood for the response variable 'y'
    y[i] ~ dnegbin(prob = prob[i], size = size)
    
   #log(mu[i]) <- beta0 + beta1 * x1[i] + beta2 * x2[i]
   # y[i] ~ dnegbin(prob = size/(size + mu[i]), size = size) 
    
  }
})

# --- 3. Prepare Data, Constants, and Initial Values ---

# Data list for Nimble
nimble_data <- list(
  y = roaches$y
)

# Constants list for Nimble
nimble_constants <- list(
  N = nrow(roaches),
  roach1 = roaches$roach1,
  treatment = roaches$treatment,
  senior = roaches$senior,
  exposure2 = roaches$exposure2
)

# Initial values for MCMC chains
# It's good practice to provide reasonable starting values.
# For coefficients, often 0 is a good start. For positive parameters like 'size',
# a small positive number or an estimate from `glm.nb` can be used.
#nimble_inits <- function() {
#  list(
#    intercept = rnorm(1, 0, 1),
#    beta_roach1 = rnorm(1, 0, 0.1),
#    beta_treatment = rnorm(1, 0, 0.5),
#    beta_senior = rnorm(1, 0, 0.5),
#    size = runif(1, 0.5, 5) # Ensure size is positive
#  )
#}

# --- 4. Compile and Run MCMC ---

# Create a Nimble model object
R_model <- nimbleModel(
  code = neg_binom_regression_code,
  constants = nimble_constants,
  data = nimble_data#,
  #inits = nimble_inits()
)

# Compile the model to C++ for speed
C_model <- compileNimble(R_model)

# Configure MCMC
# We need to monitor all parameters of interest.
mcmc_config <- configureMCMC(C_model,
                             monitors = c("intercept", "beta_roach1", "beta_treatment", "beta_senior", "size"),
                             enableWAIC = FALSE # Set to TRUE if you need WAIC, but it adds computational overhead
)

# Build the MCMC algorithm
mcmc_build <- buildMCMC(mcmc_config)

# Compile the MCMC algorithm
C_mcmc <- compileNimble(mcmc_build, project = R_model)

# Run MCMC
# Using multiple chains to check for convergence
n_iter <- 100000 # Total iterations
n_burnin <- 10000 # Burn-in period
n_chains <- 4 # Number of MCMC chains
n_thin <- 1 # Thinning interval

print(paste("Running MCMC with", n_chains, "chains, each for", n_iter, "iterations..."))
mcmc_output <- runMCMC(C_mcmc,
                       niter = n_iter,
                       nburnin = n_burnin,
                       nchains = n_chains,
                       thin = n_thin,
                       #set.seed = c(1, 2, 3), # Seeds for reproducibility across chains
                       progressBar = TRUE,
                       samplesAsCodaMCMC = TRUE # Return output as a coda::mcmc.list object
)
print("MCMC finished.")


intercept_nim<-c(mcmc_output[,"intercept"][[1]],
             mcmc_output[,"intercept"][[2]],
             mcmc_output[,"intercept"][[3]],
             mcmc_output[,"intercept"][[4]])

beta_roach1_nim<-c(mcmc_output[,"beta_roach1"][[1]],
             mcmc_output[,"beta_roach1"][[2]],
             mcmc_output[,"beta_roach1"][[3]],
             mcmc_output[,"beta_roach1"][[4]])

beta_treatment_nim<-c(mcmc_output[,"beta_treatment"][[1]],
                   mcmc_output[,"beta_treatment"][[2]],
                   mcmc_output[,"beta_treatment"][[3]],
                   mcmc_output[,"beta_treatment"][[4]])

beta_senior_nim<-c(mcmc_output[,"beta_senior"][[1]],
                      mcmc_output[,"beta_senior"][[2]],
                      mcmc_output[,"beta_senior"][[3]],
                      mcmc_output[,"beta_senior"][[4]])

size_disp_nim<-c(mcmc_output[,"size"][[1]],
                   mcmc_output[,"size"][[2]],
                   mcmc_output[,"size"][[3]],
                   mcmc_output[,"size"][[4]])


# --- 5. Analyze Results ---

par(mfrow=c(1,1))
plot(density(res_m[,"(Intercept)"]),col="green")
lines(density(res2$alpha),col="orange")
lines(density(intercept_nim),col="slateblue")

plot(density(res_m[,"roach1"]),col="green")
lines(density(res2$beta_roach1),col="orange")
lines(density(beta_roach1_nim),col="slateblue")

plot(density(res_m[,"treatment"]),col="green")
lines(density(res2$beta_treatment),col="orange")
lines(density(beta_treatment_nim),col="slateblue")

plot(density(res_m[,"senior"]),col="green")
lines(density(res2$beta_senior),col="orange")
lines(density(beta_senior_nim),col="slateblue")

plot(density(res_m[,"reciprocal_dispersion"]),col="green")
lines(density(res2$phi),col="orange")
lines(density(size_disp_nim),col="slateblue")





pdf("plot_negbin.pdf")
par(mfrow=c(1,1))
plot(density(res_m[,"(Intercept)"]),col="green")
lines(density(res2$alpha),col="orange")
lines(density(intercept_nim),col="slateblue")

plot(density(res_m[,"roach1"]),col="green")
lines(density(res2$beta_roach1),col="orange")
lines(density(beta_roach1_nim),col="slateblue")

plot(density(res_m[,"treatment"]),col="green")
lines(density(res2$beta_treatment),col="orange")
lines(density(beta_treatment_nim),col="slateblue")

plot(density(res_m[,"senior"]),col="green")
lines(density(res2$beta_senior),col="orange")
lines(density(beta_senior_nim),col="slateblue")

plot(density(res_m[,"reciprocal_dispersion"]),col="green")
lines(density(res2$phi),col="orange")
lines(density(size_disp_nim),col="slateblue")

dev.off()


# View MCMC summary statistics
#summary(mcmc_output)

# Plot MCMC traces and density plots for diagnostics
#plot(mcmc_output)




