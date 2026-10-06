// Test harness only; never used for fitting participant data.
functions {
  #include gu_stable_functions.stan
}
data {
  int<lower=0,upper=1> y;
  real<lower=0> offer;
}
parameters {
  real log_alpha;
  real log_gamma;
  real norm;
  real bias;
}
model {
  target += stable_choice_lpmf(y | log_alpha, log_gamma, norm, offer, bias);
}
