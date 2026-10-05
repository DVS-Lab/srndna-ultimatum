// Exploratory extensions of Gu-style updated-norm choice utility.
// Positive alpha/gamma and optional shared participant acceptance bias.
// Each unit is one participant/partner; its 48 offers retain chronological order.
functions {
  vector choice_logits(vector theta, row_vector offers, int model_id) {
    int T = num_elements(offers);
    vector[T] eta;
    real norm = theta[3];
    real bias = model_id == 2 ? theta[5] : 0;
    for (t in 1:T) {
      norm += theta[4] * (offers[t] - norm);
      eta[t] = bias + theta[2] * (offers[t] - theta[1] * fmax(norm - offers[t], 0));
    }
    return eta;
  }
  real partial_likelihood(array[] int units, int start, int end,
                          matrix theta, matrix offers, array[,] int choice,
                          array[,] int use_choice, int model_id) {
    real lp = 0;
    for (i in 1:size(units)) {
      int u = units[i];
      vector[cols(offers)] eta = choice_logits(theta[u]', offers[u], model_id);
      for (t in 1:cols(offers))
        if (use_choice[u,t]) lp += bernoulli_logit_lpmf(choice[u,t] | eta[t]);
    }
    return lp;
  }
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
  matrix[U,K+B] theta;
  for (u in 1:U) {
    for (k in 1:K) {
      real latent = mu[k] + sigma_subject[k] * z_subject[subject[u],k];
      for (c in 1:2)
        latent += contrast[partner[u],c] *
                  (effect[c,k] + sigma_contrast[c,k] * z_contrast[c][subject[u],k]);
      if (k <= 2) theta[u,k] = exp(latent);
      else if (k == 3) theta[u,k] = 20 * inv_logit(latent);
      else theta[u,k] = inv_logit(latent);
    }
    if (B == 1) theta[u,5] = bias_mu[1] + bias_sigma[1] * bias_z[subject[u],1];
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
  target += reduce_sum(partial_likelihood, unit_ids, 1, theta, offers,
                       choice, use_choice, model_id);
}
generated quantities {
  vector[U] train_log_lik = rep_vector(0, U);
  vector[U] heldout_log_lik = rep_vector(0, U);
  // Compact posterior predictive counts: partner x run x integer offer.
  array[3,2,11] int replicated_accept = rep_array(0, 3, 2, 11);
  array[3,2,11] real expected_accept = rep_array(0.0, 3, 2, 11);
  for (u in 1:U) {
    vector[T] eta = choice_logits(theta[u]', offers[u], model_id);
    for (t in 1:T) {
      if (observed[u,t]) {
        int b = to_int(offers[u,t]) + 1;
        real ll = bernoulli_logit_lpmf(choice[u,t] | eta[t]);
        if (use_choice[u,t]) train_log_lik[u] += ll;
        else heldout_log_lik[u] += ll;
        replicated_accept[partner[u],run[u,t],b] += bernoulli_logit_rng(eta[t]);
        expected_accept[partner[u],run[u,t],b] += inv_logit(eta[t]);
      }
    }
  }
}
