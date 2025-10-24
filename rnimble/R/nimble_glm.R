#'
#' Fit a Bayesian glm in nimble where the model was first defined and fitted using stan_glm()
#'
#' Core purpose of this function is to compare posterior estimates between rstanarm and nimble
#'
#' @param stanobj A stanfit object created from a call to stan_glm()
#' @param seed A vector of integers same length as number of chains
#' @param warmup A single integer for the burn-in in each chain
#' @param iter A single integer for total number of mcmc interations
#' @param chains A single integer for number of independent mcmc chains to be run
#'   each end of x before the mean is computed
#'
#' @return A stanfit object which is a copy of the stanjob passed and retrofitted
#'   with results from nimble
#'
#' @examples
#' nimble_glm(mystanobj,c(12345,12345), warmup=1000, iter=100000, thin=1, chains=2)
#'
#'
#'
#' @export
nimble_glm <- function(stanobj,
                       seed = c(12345),
                       warmup = 100,      # Number of warmup iterations per chain
                       iter = 1000,        # Total iterations per chain (warmup + sampling)
                       thin = 1,
                       chains = 1) {

 #useful note - if intercept is ommited then stop as suggests not centering - not implemented


  print("Hello, world!")
}
