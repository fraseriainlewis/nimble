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
data(mtcars)

# Rescale
# note for models that do internal predictor centering then need location shift back for intecept
# i.e y = a + b*(x1-mean) + c*(x2-mean)
# so want a at x1=0 and x2=0 so E(y) = mean(a) + mean(b)*-mean(x1) + mean(c)*-mean(x2) etc.
# this centering does not appear to be used in neg bin - as y is counts seem reasonable
# observe lambda*t = a + b + c but want just lambda, i.e. Y=lambda*t so lambda = Y/t
# log(lambda) = a + b + c  + log(exposure)
# P(X=x) = lambda^x exp(-lambda)/x!  lambda = lambda2*t
# 

library("rstanarm")
default_prior_test <- stan_glm(mpg ~ wt + am, data = mtcars, chains = 1)

# Estimate original model
glm1 <- glm(mpg ~ wt + am, 
            data = mtcars, family = gaussian)
# Estimate Bayesian version with stan_glm
stan_glm1 <- stan_glm(mpg ~ wt + am, data = mtcars, family = gaussian(), 
                      prior = normal(0, 2.5), 
                      prior_intercept = normal(0, 5),
                      seed = 12345,
                      warmup = 10000,      # Number of warmup iterations per chain
                      iter = 100000,        # Total iterations per chain (warmup + sampling)
                      thin = 1,
                      chains = 4)           # Thinning rate)
res_m<-as.matrix(stan_glm1)
summary(res_m[,"(Intercept)"])
summary(res_m[,"wt"])

prior_scales<-prior_summary(stan_glm1)
# get the predictor adjusted scale - stating priors explicitly so no autoscaling
beta_prior_scale<-prior_scales$prior$scale
# THE ABOVE NEEDS FIXED - if no priors are given this will be incorrect
# note - also hard coded 5 in stan below which needs fixed
#  aux is always re-scaled -next line always needed
sd_prior_scale<-prior_scales$prior_aux$adjusted_scale #need 1/sd_prior_scale for exp

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
  array[N] real<lower=0> y;         // Response variable (counts)
  vector[N] wt;           // Continuous predictor
  vector[N] am;    // Binary predictor
  //Hyperparameters
  array[M] real<lower=0>rescaled_sd;// standard dev for predictors

}

transformed data {
  vector[N] wt_centered;
  real mean_wt=mean(wt);
  vector[N] am_centered;
  real mean_am=mean(am);
  wt_centered = wt - mean_wt;  // Center in transformed data block
  am_centered = am - mean_am;
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
                beta_wt * wt_centered[i] +
                beta_am * am_centered[i] ; 
  }
}

model {
  // --- Priors ---
  // Weakly informative priors for coefficients and intercept
  alpha ~ normal(0, 5.0);           // Prior for intercept
  beta_wt ~ normal(0,rescaled_sd[1] );     // Prior for roach1 coefficient
  beta_am ~ normal(0, rescaled_sd[2]);  // Prior for treatment coefficient
  phi ~ exponential(rescaled_sd[3]); // this is actually 1/ rescaled scale

  // --- Likelihood ---
  // Negative Binomial likelihood, using the log-link function for the mean
  y ~  normal(mu, phi);
}

generated quantities {
  // Can include posterior predictions or log-likelihood here if desired for model checking
  real intercept_0;
  intercept_0=alpha + beta_wt*-mean_wt + beta_am*-mean_am; 
  
  
  
}
"

# --- 3. Prepare data for Stan ---
# The data needs to be provided as a list for rstan::stan()
stan_data <- list(
  N = nrow(mtcars),
  M = 3, # number of passed hyperpriors
  rescaled_sd=c(beta_prior_scale,1/sd_prior_scale),# 1/ as prior uses rate
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
  warmup = 10000,      # Number of warmup iterations per chain
  iter = 100000,        # Total iterations per chain (warmup + sampling)
  thin = 1,           # Thinning rate
  seed = 12345,          # For reproducibility
  control = list(adapt_delta = 0.95, max_treedepth = 15) # Adjust for sampling issues if needed
)

# --- 5. Extract main parameters and produce density plots ---
res2<-extract(fit,par=c("alpha","beta_wt"," beta_am","phi","intercept_0"))

#par(mfrow=c(1,2))
#plot(density(res_m[,"(Intercept)"]))
#lines(density(res2$alpha),col="red")

#plot(density(res_m[,"roach1"]))
#lines(density(res2$beta_roach1),col="blue")

if(FALSE){
pdf("plot_gaus.pdf")
par(mfrow=c(1,1))
plot(density(res_m[,"(Intercept)"]),col="green")
lines(density(res2$intercept_0),col="orange")


plot(density(res_m[,"wt"]),col="green")
lines(density(res2$beta_wt),col="orange")

plot(density(res_m[,"am"]),col="green")
lines(density(res2$beta_am),col="orange")

plot(density(res_m[,"sigma"]),col="green")
lines(density(res2$phi),col="orange")
dev.off()
}

################################################################################

# Load necessary libraries
library(nimble)
library(coda) # For MCMC diagnostics and plotting


# --- Define the Nimble Model Code ---
# This block defines the statistical model using Nimble's DSL.
# The negative binomial distribution in Nimble (dnegbin) is parameterized
# by `prob` and `size`. Its mean is `size * (1-prob) / prob`.
# We relate this mean to a linear predictor `log_lambda`.
gaussian_regression_code <- nimbleCode({
  # Priors for regression coefficients and intercept (weak priors)
  # Using normal distributions with large standard deviations (small precision/tau)
  intercept ~ dnorm(0, sd = 5) # Weakly informative prior for intercept
  beta_wt ~ dnorm(0, sd = hypers[1]) # Weakly informative prior for roach1 coefficient
  beta_am ~ dnorm(0, sd = hypers[2]) # Weakly informative prior for treatment coefficient
  
  # Prior for the Negative Binomial dispersion parameter (size)
  # 'size' must be positive. A Gamma distribution is a common choice for scale parameters.
  # dgamma(shape, rate) with small shape and rate implies a weak prior.
  #size ~ dgamma(0.01, 0.01) # Weakly informative prior for dispersion parameter
  sd ~ dexp(hypers[3])
  
  mean_wt <- mean(wt[])
  mean_am <- mean(am[])
  wt_centered[1:N]<-wt[]-mean_wt
  am_centered[1:N]<-am[]-mean_am
  
  # Likelihood for each observation
  for (i in 1:N) {
    # Linear predictor on the log scale (log_lambda_expected)
    # exposure2 is an offset, so log(exposure2[i]) is added directly
    # with a coefficient fixed at 1.
    mu[i] <- intercept +
      beta_wt * wt_centered[i] +
      beta_am * am_centered[i]

    # Convert lambda and size to 'prob' parameter for dnegbin
    # dnegbin(prob, size) has mean = size * (1 - prob) / prob
    # So, lambda[i] = size * (1 - prob[i]) / prob[i]
    # Rearranging for prob[i]: prob[i] = size / (size + lambda[i])
    # An equivalent form is prob[i] = 1 / (1 + lambda[i] / size)
    #prob[i] <- 1 / (1 + lambda[i] / size)
    
    # Negative Binomial likelihood for the response variable 'y'
    y[i] ~ dnorm(mu[i], sd=sd)
  }
    intercept_0 <-intercept + beta_wt*-mean_wt + beta_am*-mean_am;
    
    #log(mu[i]) <- beta0 + beta1 * x1[i] + beta2 * x2[i]
    # y[i] ~ dnegbin(prob = size/(size + mu[i]), size = size) 
    
  
})

# --- 3. Prepare Data, Constants, and Initial Values ---

# Data list for Nimble
nimble_data <- list(
  y = mtcars$mpg
)

# Constants list for Nimble
nimble_constants <- list(
  N = nrow(mtcars),
  wt = mtcars$wt,
  am = mtcars$am,
  hypers=c(beta_prior_scale,1/sd_prior_scale)
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
  code = gaussian_regression_code,
  constants = nimble_constants,
  data = nimble_data#,
  #inits = nimble_inits()
)

# Compile the model to C++ for speed
C_model <- compileNimble(R_model)

# Configure MCMC
# We need to monitor all parameters of interest.
mcmc_config <- configureMCMC(C_model,
                             monitors = c("intercept", "beta_wt", "beta_am", "sd","intercept_0"),
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


intercept_nim<-c(mcmc_output[,"intercept_0"][[1]],
                 mcmc_output[,"intercept_0"][[2]],
                 mcmc_output[,"intercept_0"][[3]],
                 mcmc_output[,"intercept_0"][[4]])

beta_wt_nim<-c(mcmc_output[,"beta_wt"][[1]],
                   mcmc_output[,"beta_wt"][[2]],
                   mcmc_output[,"beta_wt"][[3]],
                   mcmc_output[,"beta_wt"][[4]])

beta_am_nim<-c(mcmc_output[,"beta_am"][[1]],
                      mcmc_output[,"beta_am"][[2]],
                      mcmc_output[,"beta_am"][[3]],
                      mcmc_output[,"beta_am"][[4]])

size_sd_nim<-c(mcmc_output[,"sd"][[1]],
                 mcmc_output[,"sd"][[2]],
                 mcmc_output[,"sd"][[3]],
                 mcmc_output[,"sd"][[4]])


# --- 5. Analyze Results ---
pdf("plot_gaus.pdf")
par(mfrow=c(1,1))
plot(density(res_m[,"(Intercept)"]),col="green")
lines(density(res2$intercept_0),col="orange")
lines(density(intercept_nim),col="slateblue")


plot(density(res_m[,"wt"]),col="green")
lines(density(res2$beta_wt),col="orange")
lines(density(beta_wt_nim),col="slateblue")

plot(density(res_m[,"am"]),col="green")
lines(density(res2$beta_am),col="orange")
lines(density(beta_am_nim),col="slateblue")

plot(density(res_m[,"sigma"]),col="green")
lines(density(res2$phi),col="orange")
lines(density(size_sd_nim),col="slateblue")
dev.off()




