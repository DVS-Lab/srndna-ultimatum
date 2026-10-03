// Gu et al. (2015), doi:10.1523/JNEUROSCI.2906-14.2015.
// Each unit is one participant/partner; its 48 offers retain chronological order.
functions {
  vector choice_logits(vector theta, row_vector offers, int model_id) {
    int T = num_elements(offers);
    vector[T] eta;
    if (model_id == 3) {
      for (t in 1:T) eta[t] = theta[1] + theta[2] * (offers[t] - 5);
    } else {
      real norm = theta[3];
      for (t in 1:T) {
        real at_choice = norm;
        if (model_id != 2) norm += theta[4] * (offers[t] - norm);
        if (model_id != 4) at_choice = norm;
        eta[t] = theta[2] * (offers[t] - theta[1] * fmax(at_choice - offers[t], 0));
      }
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
  int<lower=2,upper=4> K;
  int<lower=1,upper=4> model_id; // RW updated, static norm, logistic, RW prior norm
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
  int<lower=0,upper=1> A; // 0: age-blind; 1: centered older-group indicator
  matrix[S,A] age_design;
  vector<lower=0>[K] age_sd;
}
parameters {
  vector[K] mu;
  matrix[2,K] effect;
  vector<lower=0>[K] sigma_subject;
  matrix<lower=0>[2,K] sigma_contrast;
  matrix[S,K] z_subject;
  array[2] matrix[S,K] z_contrast;
  matrix[A,K] age_beta;
}
transformed parameters {
  matrix[U,K] theta;
  for (u in 1:U) {
    for (k in 1:K) {
      real latent = mu[k] + sigma_subject[k] * z_subject[subject[u],k];
      if (A == 1) latent += age_design[subject[u],1] * age_beta[1,k];
      for (c in 1:2)
        latent += contrast[partner[u],c] *
                  (effect[c,k] + sigma_contrast[c,k] * z_contrast[c][subject[u],k]);
      if (model_id == 3) {
        // Preserve the MLE comparator's bounds; nearly linear near zero.
        if (k == 1) theta[u,k] = 30 * tanh(latent / 30);
        else theta[u,k] = 10 * tanh(latent / 10);
      } else {
        theta[u,k] = inv_logit(latent);
        if (k == 3) theta[u,k] *= 20;
      }
    }
  }
}
model {
  mu ~ normal(0, mu_sd);
  sigma_subject ~ normal(0, subject_sd);
  to_vector(z_subject) ~ std_normal();
  if (A == 1) age_beta[1]' ~ normal(0, age_sd);
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
