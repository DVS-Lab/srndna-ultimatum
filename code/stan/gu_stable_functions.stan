// Algebraically unchanged logit: bias + gamma*(offer-alpha*max(norm-offer,0)).
// Work with log(alpha) and log(gamma) until the final likelihood calculation.
real stable_choice_lpmf(int y, real log_alpha, real log_gamma,
                        real norm, real offer, real bias) {
  real positive = negative_infinity();
  real negative = negative_infinity();
  if (offer > 0) positive = log_gamma + log(offer);
  if (norm > offer) negative = log_gamma + log_alpha + log(norm-offer);
  // Cancel the utility terms before adding a small bias. Combining that bias
  // into a huge log term first would lose it to rounding.
  if (positive == negative && positive != negative_infinity()) {
    if (positive > log(1.7976931348623157e308))
      reject("exact cancellation has unrepresentable derivatives");
    return bernoulli_logit_lpmf(y | (exp(positive)-exp(negative)) + bias);
  }
  if (bias > 0) positive = log_sum_exp(positive, log(bias));
  if (bias < 0) negative = log_sum_exp(negative, log(-bias));
  // Retain the derivative w.r.t. bias at zero, and ordinary derivatives at
  // exact cancellation. The direct branch is safe below exp(700).
  if (fmax(positive, negative) < 700) {
    real positive_without_bias = offer > 0 ? exp(log_gamma + log(offer)) : 0;
    real negative_without_bias = norm > offer ? exp(log_gamma + log_alpha + log(norm-offer)) : 0;
    return bernoulli_logit_lpmf(y | bias + positive_without_bias - negative_without_bias);
  }
  // For extreme, distinct log terms, |eta| is enormous: matching outcomes
  // have log probability rounded to zero; opposing outcomes have -|eta|.
  // This is floating-point asymptotic evaluation, not logit clipping.
  if (positive == negative) {
    if (positive > log(1.7976931348623157e308))
      reject("exact cancellation has unrepresentable derivatives");
    return bernoulli_logit_lpmf(y | exp(positive)-exp(negative) + (bias == 0 ? bias : 0));
  } else {
    int positive_eta = positive > negative;
    real log_magnitude = positive_eta ? log_diff_exp(positive, negative)
                                     : log_diff_exp(negative, positive);
    if (y == positive_eta) return 0;
    if (log_magnitude > log(1.7976931348623157e308)) return negative_infinity();
    return -exp(log_magnitude);
  }
}

real stable_partial_likelihood(array[] int units, int start, int end,
                              matrix latent, vector bias, matrix offers,
                              array[,] int choice, array[,] int use_choice) {
  real lp = 0;
  for (i in 1:size(units)) {
    int u = units[i];
    real norm = 20 * inv_logit(latent[u,3]);
    real epsilon = inv_logit(latent[u,4]);
    for (t in 1:cols(offers)) {
      norm += epsilon * (offers[u,t] - norm);
      if (use_choice[u,t])
        lp += stable_choice_lpmf(choice[u,t] | latent[u,1], latent[u,2], norm, offers[u,t], bias[u]);
    }
  }
  return lp;
}
