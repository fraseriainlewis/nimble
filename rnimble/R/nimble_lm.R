#'
#' Fit a Bayesian glm in nimble where the model was first defined and fitted using stan_glm()
#'
#' Core purpose of this function is to compare posterior estimates between rstanarm and nimble
#'
#' @param stanobj A stanfit object created from a call to stan_glm()
#' @return A stanfit object which is a copy of the stanjob passed and retrofitted
#'   with results from nimble
#'
#'
#' @export
build_lm <- function(stanobj) {

   # code = str2expression((string form of nimble model))
   #nimbleModelexpr<-build_lm(stanobj); #decontruct object and build full model definition


  print("Hello, world2!")
}
