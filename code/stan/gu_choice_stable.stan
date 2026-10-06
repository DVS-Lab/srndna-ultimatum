// Same hierarchy/priors and choice equation as gu_choice_extensions.stan.
// Exponentiated reporting parameters are generated only after target evaluation.
functions {
  #include gu_stable_functions.stan
}
data {
  int<lower=1> S;
  int<lower=1> U;
  int<lower=1> T;
  int<lower=4,upper=4> K;
  int<lower=0,upper=1> B;
  real<lower=0> bias_mu_sd;
  real<lower=0> bias_subject_sd;
  int<lower=1,upper=2> model_id; // positive support, positive support plus bias
  array[U] int<lower=1,upper=S> subject;
  array[U] int<lower=1,upper=3> partner; // computer, similar, dissimilar
  array[U] int<lower=1,upper=U> unit_ids;
  matrix<lower=0,upper=10>[U,T] offers;
  array[U,T] int<lower=0,upper=1> choice;
  array[U,T] int<lower=0,upper=1> observed;
  array[U,T] int<lower=0,upper=1> use_choice;
  array[U,T] int<lower=1,upper=2> run;
  matrix[3,2] contrast;
  vector<lower=0>[K] mu_sd;
  vector<lower=0>[K] effect_sd;
  vector<lower=0>[K] subject_sd;
  vector<lower=0>[K] contrast_sd;
  int<lower=0,upper=0> A; // extension is strictly age-blind
}
transformed data {
  if (B != model_id - 1 || A != 0) reject("extension mode/bias/age mismatch");
}
parameters {
  vector[B] bias_mu;
  vector<lower=0>[B] bias_sigma;
  matrix[S,B] bias_z;
  vector[K] mu;
  matrix[2,K] effect;
  vector<lower=0>[K] sigma_subject;
  matrix<lower=0>[2,K] sigma_contrast;
  matrix[S,K] z_subject;
  array[2] matrix[S,K] z_contrast;
}
transformed parameters {
  matrix[U,K] latent;
  vector[U] unit_bias;
  for (u in 1:U) {
    for (k in 1:K) {
      latent[u,k] = mu[k] + sigma_subject[k] * z_subject[subject[u],k];
      for (c in 1:2)
        latent[u,k] += contrast[partner[u],c] *
          (effect[c,k] + sigma_contrast[c,k] * z_contrast[c][subject[u],k]);
    }
    unit_bias[u] = B == 1 ? bias_mu[1] + bias_sigma[1] * bias_z[subject[u],1] : 0;
  }
}

model {
  bias_mu ~ normal(0, bias_mu_sd);
  bias_sigma ~ normal(0, bias_subject_sd);
  to_vector(bias_z) ~ std_normal();
  mu ~ normal(0, mu_sd);
  sigma_subject ~ normal(0, subject_sd);
  to_vector(z_subject) ~ std_normal();
  for (c in 1:2) {
    effect[c]' ~ normal(0, effect_sd);
    sigma_contrast[c]' ~ normal(0, contrast_sd);
    to_vector(z_contrast[c]) ~ std_normal();
  }
  target += reduce_sum(stable_partial_likelihood, unit_ids, 1, latent, unit_bias, offers,
                       choice, use_choice);
}
generated quantities {
  matrix[U,K+B] theta;
  vector[U] train_log_lik = rep_vector(0,U);
  vector[U] heldout_log_lik = rep_vector(0,U);
  array[3,2,11] int replicated_accept = rep_array(0,3,2,11);
  array[3,2,11] real expected_accept = rep_array(0.0,3,2,11);
  for (u in 1:U) {
    real norm = 20 * inv_logit(latent[u,3]);
    real epsilon = inv_logit(latent[u,4]);
    theta[u,1] = exp(latent[u,1]);
    theta[u,2] = exp(latent[u,2]);
    theta[u,3] = norm;
    theta[u,4] = epsilon;
    if (B == 1) theta[u,5] = unit_bias[u];
    for (t in 1:T) {
      norm += epsilon * (offers[u,t]-norm);
      if (observed[u,t]) {
        int b = to_int(offers[u,t])+1;
        real ll = stable_choice_lpmf(choice[u,t] | latent[u,1], latent[u,2], norm, offers[u,t], unit_bias[u]);
        real p = exp(stable_choice_lpmf(1 | latent[u,1], latent[u,2], norm, offers[u,t], unit_bias[u]));
        if (use_choice[u,t]) train_log_lik[u] += ll;
        else heldout_log_lik[u] += ll;
        replicated_accept[partner[u],run[u,t],b] += bernoulli_rng(p);
        expected_accept[partner[u],run[u,t],b] += p;
      }
    }
  }
}
