#' Add together two numbers
#'
#' @returns The sum of `x` and `y`
#' @export
#' @examples
#'
#' test_call_to_tfd()
script_neg_bin <- function() {
  print("calling....\n")
  a<-10.0
  print(tfd_bernoulli(probs=0.5)%>%tfd_sample(5L))
  mystring<-"
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

num_schools = 8  # number of schools
treatment_effects = np.array(
    [28, 8, -3, 7, -1, 1, 18, 12], dtype=np.float32)  # treatment effects
treatment_stddevs = np.array(
    [15, 10, 16, 11, 9, 11, 10, 18], dtype=np.float32)  # treatment SE


fromr= r.a + 10
"
  py_run_string(mystring)
  cat("got=",py$fromr,"\n")










  }
