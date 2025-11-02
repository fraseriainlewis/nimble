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

..onLoad <- function(libname, pkgname) {
  reticulate::py_require("tensorflow")
  reticulate::py_require("tf_keras")
  reticulate::py_require("tensorflow_probability")
}

#' Add together two numbers
#'
#' @returns The sum of `x` and `y`
#' @export
#' @examples
#' test_call_to_tfd()
  test_call_to_tfd <- function() {
   print("calling....\n")
    return(tfd_bernoulli(probs=0.5)%>%tfd_sample(5L))
}
