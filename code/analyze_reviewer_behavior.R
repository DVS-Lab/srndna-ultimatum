#!/usr/bin/env Rscript

# Reproducible behavioral analyses requested during peer review.
#
# This script keeps submitted and revision analyses explicitly labeled, writes
# compact tables/figure source data, and does not overwrite legacy artifacts.

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(ggplot2)
})

options(contrasts = c("contr.treatment", "contr.poly"))

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
if (length(script_arg) == 1) {
  script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
  project_root <- dirname(dirname(script_path))
} else {
  project_root <- normalizePath(".", mustWork = TRUE)
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) > 0) {
  root_arg <- args[grepl("^--root=", args)]
  if (length(root_arg) == 1) project_root <- normalizePath(sub("^--root=", "", root_arg), mustWork = TRUE)
}

data_dir <- file.path(project_root, "behavioral_analyses", "data")
event_dir <- file.path(project_root, "results", "reviewer", "tables")
table_dir <- event_dir
private_dir <- file.path(project_root, "results", "reviewer", "private")
figure_dir <- file.path(project_root, "results", "reviewer", "figures")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(private_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

write_tsv <- function(x, path) {
  write.table(x, path, sep = "\t", row.names = FALSE, quote = FALSE, na = "NA")
}

wald_table <- function(model, model_id, analysis_status, n_subjects, n_trials) {
  coef_table <- as.data.frame(coef(summary(model)))
  coef_table$term <- rownames(coef_table)
  rownames(coef_table) <- NULL
  stat_name <- if ("z value" %in% names(coef_table)) "z value" else "t value"
  p_name <- grep("^Pr\\(", names(coef_table), value = TRUE)
  ci <- suppressMessages(confint(model, parm = "beta_", method = "Wald"))
  messages <- model@optinfo$conv$lme4$messages
  data.frame(
    model_id = model_id,
    analysis_status = analysis_status,
    term = coef_table$term,
    estimate = coef_table$Estimate,
    std_error = coef_table$`Std. Error`,
    statistic = coef_table[[stat_name]],
    p_value = if (length(p_name) == 1) coef_table[[p_name]] else NA_real_,
    conf_low = ci[, 1],
    conf_high = ci[, 2],
    n_participants = n_subjects,
    n_trials = n_trials,
    singular = isSingular(model, tol = 1e-4),
    convergence_message = if (is.null(messages)) NA_character_ else paste(messages, collapse = " | "),
    check.names = FALSE
  )
}

read_csv_clean <- function(path) read.csv(path, check.names = FALSE, fileEncoding = "UTF-8-BOM")

read_group_fsf_ev <- function(path, ev_index) {
  lines <- readLines(path, warn = FALSE)
  input_lines <- grep("^set feat_files\\([0-9]+\\) ", lines, value = TRUE)
  input_index <- as.integer(sub("^set feat_files\\(([0-9]+)\\).*$", "\\1", input_lines))
  participant <- sub('^.*(sub-[0-9]+).*$','\\1', input_lines)
  value_pattern <- sprintf("^set fmri\\(evg[0-9]+\\.%d\\) ", ev_index)
  value_lines <- grep(value_pattern, lines, value = TRUE)
  value_index <- as.integer(sub("^set fmri\\(evg([0-9]+)\\..*$", "\\1", value_lines))
  value <- as.numeric(sub(value_pattern, "", value_lines))
  if (length(input_index) == 0 || length(input_index) != length(value_index)) {
    stop(sprintf("incomplete EV %d in %s", ev_index, path))
  }
  values <- value[match(input_index, value_index)]
  if (anyNA(values) || anyDuplicated(participant)) stop(sprintf("invalid EV %d in %s", ev_index, path))
  data.frame(subjID = participant, value = values, stringsAsFactors = FALSE)
}

participant_rt_summary <- function(x, suffix) {
  x <- subset(x, missed == 0 & is.finite(response_time))
  means <- aggregate(response_time ~ subjID, data = x, FUN = mean)
  medians <- aggregate(response_time ~ subjID, data = x, FUN = median)
  names(means)[2] <- paste0("response_time_mean_", suffix)
  names(medians)[2] <- paste0("response_time_median_", suffix)
  merge(means, medians, by = "subjID")
}

participants <- read_csv_clean(file.path(data_dir, "participant_L3_47.csv"))
names(participants)[1] <- "subjID"
participants$age_group <- factor(
  ifelse(participants$younger == 1, "younger", "older"),
  levels = c("younger", "older")
)

trials_path <- file.path(data_dir, "all_trials_brains.csv")
trials_arg <- args[grepl("^--trials=", args)]
if (length(trials_arg) == 1) {
  candidate <- sub("^--trials=", "", trials_arg)
  if (!grepl("^/", candidate)) candidate <- file.path(project_root, candidate)
  trials_path <- normalizePath(candidate, mustWork = TRUE)
}
trials <- read_csv_clean(trials_path)
trials <- merge(trials, participants[, c("subjID", "age_group")], by = "subjID")
human <- subset(trials, missed == 0 & human == 1)
human$accept <- as.integer(as.character(human$accept))
human$similarity <- factor(ifelse(human$ingroup == 1, "similar", "dissimilar"), levels = c("dissimilar", "similar"))
human$similarity_num <- as.integer(human$similarity == "similar")
human$offer_c <- human$offer - mean(human$offer)
human$offer_3 <- human$offer - 3
human$subjID <- factor(human$subjID)
human$age_group <- factor(human$age_group, levels = c("younger", "older"))

stopifnot(nlevels(human$subjID) == 47, nrow(human) == 4438)

glmer_control <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

# Primary refit: the intended maximal random-effects model now converges.
primary_maximal <- glmer(
  accept ~ offer_c * age_group * similarity + (1 + offer_c | subjID),
  data = human,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
)
primary_intercept <- glmer(
  accept ~ offer_c * age_group * similarity + (1 | subjID),
  data = human,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
)

primary_table <- rbind(
  wald_table(primary_maximal, "primary_maximal", "revision-event-corrected-refit", 47, nrow(human)),
  wald_table(primary_intercept, "random_intercept_robustness", "revision-event-corrected-robustness", 47, nrow(human))
)
write_tsv(primary_table, file.path(table_dir, "primary_acceptance_models.tsv"))

# Cross-optimizer agreement for the maximal model. The primary fit uses an
# inline control expression so lme4::allFit can safely replace the optimizer.
optimizer_fits <- suppressWarnings(suppressMessages(allFit(primary_maximal, verbose = FALSE)))
optimizer_rows <- lapply(names(optimizer_fits), function(name) {
  fit <- optimizer_fits[[name]]
  if (!inherits(fit, "merMod")) {
    return(data.frame(optimizer = name, converged = FALSE, singular = NA, estimate = NA, std_error = NA, log_likelihood = NA))
  }
  table <- coef(summary(fit))
  term <- "offer_c:age_groupolder:similaritysimilar"
  messages <- fit@optinfo$conv$lme4$messages
  data.frame(
    optimizer = name,
    converged = is.null(messages),
    singular = isSingular(fit, tol = 1e-4),
    estimate = table[term, "Estimate"],
    std_error = table[term, "Std. Error"],
    log_likelihood = as.numeric(logLik(fit))
  )
})
write_tsv(do.call(rbind, optimizer_rows), file.path(table_dir, "primary_optimizer_diagnostics.tsv"))

# Submitted two-model operationalization, rederived before comparison.
similar_fit <- glmer(accept ~ offer + (1 + offer | subjID), data = subset(human, similarity == "similar"), family = binomial, control = glmer_control)
dissimilar_fit <- glmer(accept ~ offer + (1 + offer | subjID), data = subset(human, similarity == "dissimilar"), family = binomial, control = glmer_control)

conditional_offer_slopes <- function(model) {
  values <- coef(model)$subjID$offer
  data.frame(subjID = rownames(coef(model)$subjID), value = values, stringsAsFactors = FALSE)
}
similar_slopes <- conditional_offer_slopes(similar_fit)
dissimilar_slopes <- conditional_offer_slopes(dissimilar_fit)
names(similar_slopes)[2] <- "similar_offer_slope_rederived"
names(dissimilar_slopes)[2] <- "dissimilar_offer_slope_rederived"
rederived <- merge(similar_slopes, dissimilar_slopes, by = "subjID")
rederived$corrected_separate_model_metric <- rederived$similar_offer_slope_rederived - rederived$dissimilar_offer_slope_rederived

# The submitted activation norm proxy was the difference between participant
# random intercepts from separate partner-condition fits using offer - 3. It
# is not the later intercept/slope threshold metric retained in Jen's history.
conditional_random_intercepts <- function(model) {
  effects <- ranef(model)$subjID
  values <- effects[, "(Intercept)"]
  data.frame(subjID = rownames(effects), value = as.numeric(values), stringsAsFactors = FALSE)
}
norm_similar_fit <- glmer(
  accept ~ offer_3 + (1 + offer_3 | subjID),
  data = subset(human, similarity == "similar"),
  family = binomial,
  control = glmer_control
)
norm_dissimilar_fit <- glmer(
  accept ~ offer_3 + (1 + offer_3 | subjID),
  data = subset(human, similarity == "dissimilar"),
  family = binomial,
  control = glmer_control
)
similar_intercepts <- conditional_random_intercepts(norm_similar_fit)
dissimilar_intercepts <- conditional_random_intercepts(norm_dissimilar_fit)
names(similar_intercepts)[2] <- "similar_random_intercept_rederived"
names(dissimilar_intercepts)[2] <- "dissimilar_random_intercept_rederived"
norm_proxy <- merge(similar_intercepts, dissimilar_intercepts, by = "subjID")
norm_proxy$corrected_norm_proxy <- (
  norm_proxy$similar_random_intercept_rederived -
    norm_proxy$dissimilar_random_intercept_rederived
)

# Refit the same two models to the submitted event table as a calibration
# bridge. This separates ordinary software/refit differences from the effect
# of replacing sub-144's duplicated behavioral labels.
historical_human <- subset(historical_trials <- read_csv_clean(file.path(data_dir, "all_trials_brains.csv")), missed == 0 & human == 1)
historical_human <- subset(historical_human, subjID %in% participants$subjID)
historical_human$accept <- as.integer(as.character(historical_human$accept))
historical_human$similarity <- factor(
  ifelse(historical_human$ingroup == 1, "similar", "dissimilar"),
  levels = c("dissimilar", "similar")
)
historical_human$offer_3 <- historical_human$offer - 3
historical_human$subjID <- factor(historical_human$subjID)
historical_similar_fit <- glmer(
  accept ~ offer + (1 + offer | subjID),
  data = subset(historical_human, similarity == "similar"),
  family = binomial,
  control = glmer_control
)
historical_dissimilar_fit <- glmer(
  accept ~ offer + (1 + offer | subjID),
  data = subset(historical_human, similarity == "dissimilar"),
  family = binomial,
  control = glmer_control
)
historical_similar_slopes <- conditional_offer_slopes(historical_similar_fit)
historical_dissimilar_slopes <- conditional_offer_slopes(historical_dissimilar_fit)
names(historical_similar_slopes)[2] <- "historical_refit_similar_slope"
names(historical_dissimilar_slopes)[2] <- "historical_refit_dissimilar_slope"
historical_metrics <- merge(historical_similar_slopes, historical_dissimilar_slopes, by = "subjID")
historical_metrics$historical_refit_sensitivity <- (
  historical_metrics$historical_refit_similar_slope -
    historical_metrics$historical_refit_dissimilar_slope
)
historical_norm_similar_fit <- glmer(
  accept ~ offer_3 + (1 + offer_3 | subjID),
  data = subset(historical_human, similarity == "similar"),
  family = binomial,
  control = glmer_control
)
historical_norm_dissimilar_fit <- glmer(
  accept ~ offer_3 + (1 + offer_3 | subjID),
  data = subset(historical_human, similarity == "dissimilar"),
  family = binomial,
  control = glmer_control
)
historical_similar_intercepts <- conditional_random_intercepts(historical_norm_similar_fit)
historical_dissimilar_intercepts <- conditional_random_intercepts(historical_norm_dissimilar_fit)
names(historical_similar_intercepts)[2] <- "historical_refit_similar_random_intercept"
names(historical_dissimilar_intercepts)[2] <- "historical_refit_dissimilar_random_intercept"
historical_metrics <- merge(historical_metrics, historical_similar_intercepts, by = "subjID")
historical_metrics <- merge(historical_metrics, historical_dissimilar_intercepts, by = "subjID")
historical_metrics$historical_refit_norm_proxy <- (
  historical_metrics$historical_refit_similar_random_intercept -
    historical_metrics$historical_refit_dissimilar_random_intercept
)

submitted <- read_csv_clean(file.path(data_dir, "in_out_sensitivity_indiv_logit.csv"))
submitted <- submitted[, c("subjID", "in_out_indiv_logit")]
names(submitted)[2] <- "submitted_metric_tracked"
sensitivity <- merge(rederived, submitted, by = "subjID")
sensitivity <- merge(sensitivity, participants[, c("subjID", "age_group")], by = "subjID")

# Recover the exact submitted group-design columns from the compact production
# FSFs. The two group-specific columns sum to the participant's centered value.
ecn_fsf <- file.path(
  project_root, "results", "reviewer", "production_audits",
  "ecn-sensitivity", "design", "design.fsf"
)
activation_fsf <- file.path(
  project_root, "results", "reviewer", "production_audits",
  "activation", "design", "design.fsf"
)
submitted_ecn_y <- read_group_fsf_ev(ecn_fsf, 7)
submitted_ecn_o <- read_group_fsf_ev(ecn_fsf, 8)
submitted_norm_y <- read_group_fsf_ev(activation_fsf, 7)
submitted_norm_o <- read_group_fsf_ev(activation_fsf, 8)
submitted_rt <- read_group_fsf_ev(activation_fsf, 4)
names(submitted_ecn_y)[2] <- "submitted_sensitivity_young"
names(submitted_ecn_o)[2] <- "submitted_sensitivity_old"
names(submitted_norm_y)[2] <- "submitted_norm_young"
names(submitted_norm_o)[2] <- "submitted_norm_old"
names(submitted_rt)[2] <- "submitted_group_rt"

corrected_covariates <- participants[, c("subjID", "younger", "older", "ismale", "tsnr", "fd_mean", "RT")]
corrected_covariates <- merge(corrected_covariates, rederived, by = "subjID")
corrected_covariates <- merge(corrected_covariates, norm_proxy, by = "subjID")
corrected_covariates <- merge(corrected_covariates, historical_metrics, by = "subjID")
for (x in list(submitted_ecn_y, submitted_ecn_o, submitted_norm_y, submitted_norm_o, submitted_rt)) {
  corrected_covariates <- merge(corrected_covariates, x, by = "subjID")
}
corrected_covariates$submitted_sensitivity_centered <- (
  corrected_covariates$submitted_sensitivity_young +
    corrected_covariates$submitted_sensitivity_old
)
corrected_covariates$submitted_norm_centered <- (
  corrected_covariates$submitted_norm_young +
    corrected_covariates$submitted_norm_old
)
corrected_covariates$corrected_sensitivity_centered <- (
  corrected_covariates$corrected_separate_model_metric -
    mean(corrected_covariates$corrected_separate_model_metric)
)
corrected_covariates$corrected_norm_centered <- (
  corrected_covariates$corrected_norm_proxy -
    mean(corrected_covariates$corrected_norm_proxy)
)
corrected_covariates$historical_refit_sensitivity_centered <- (
  corrected_covariates$historical_refit_sensitivity -
    mean(corrected_covariates$historical_refit_sensitivity)
)
corrected_covariates$historical_refit_norm_centered <- (
  corrected_covariates$historical_refit_norm_proxy -
    mean(corrected_covariates$historical_refit_norm_proxy)
)
corrected_covariates$corrected_sensitivity_young <- corrected_covariates$corrected_sensitivity_centered * corrected_covariates$younger
corrected_covariates$corrected_sensitivity_old <- corrected_covariates$corrected_sensitivity_centered * corrected_covariates$older
corrected_covariates$corrected_norm_young <- corrected_covariates$corrected_norm_centered * corrected_covariates$younger
corrected_covariates$corrected_norm_old <- corrected_covariates$corrected_norm_centered * corrected_covariates$older

historical_rt <- participant_rt_summary(historical_trials, "submitted_events")
corrected_rt <- participant_rt_summary(trials, "corrected_events")
corrected_covariates <- merge(corrected_covariates, historical_rt, by = "subjID")
corrected_covariates <- merge(corrected_covariates, corrected_rt, by = "subjID")
corrected_covariates <- corrected_covariates[match(participants$subjID, corrected_covariates$subjID), ]
stopifnot(
  identical(as.character(corrected_covariates$subjID), as.character(participants$subjID)),
  nrow(corrected_covariates) == 47,
  all(is.finite(corrected_covariates$corrected_sensitivity_centered)),
  all(is.finite(corrected_covariates$corrected_norm_centered))
)
write_tsv(corrected_covariates, file.path(private_dir, "corrected_l3_covariates.tsv"))
write_tsv(
  corrected_covariates[, c(
    "subjID",
    "corrected_sensitivity_young",
    "corrected_sensitivity_old",
    "corrected_norm_young",
    "corrected_norm_old"
  )],
  file.path(table_dir, "l3_event_corrected_covariates.tsv")
)

covariate_columns <- list(
  fairness_sensitivity = c("submitted_sensitivity_centered", "corrected_sensitivity_centered"),
  fairness_norm_proxy = c("submitted_norm_centered", "corrected_norm_centered")
)
covariate_summary <- do.call(rbind, Map(
  function(covariate_id, columns) {
    submitted_values <- corrected_covariates[[columns[1]]]
    corrected_values <- corrected_covariates[[columns[2]]]
    target <- corrected_covariates$subjID == "sub-144"
    data.frame(
      covariate = covariate_id,
      submitted_corrected_correlation = cor(submitted_values, corrected_values),
      maximum_absolute_difference = max(abs(submitted_values - corrected_values)),
      participants_changed_gt_1e_8 = sum(abs(submitted_values - corrected_values) > 1e-8),
      sub144_submitted_centered = submitted_values[target],
      sub144_corrected_centered = corrected_values[target]
    )
  },
  names(covariate_columns),
  covariate_columns
))
refit_columns <- list(
  fairness_sensitivity = c(
    "submitted_sensitivity_centered",
    "historical_refit_sensitivity_centered",
    "corrected_sensitivity_centered"
  ),
  fairness_norm_proxy = c(
    "submitted_norm_centered",
    "historical_refit_norm_centered",
    "corrected_norm_centered"
  )
)
covariate_summary$production_historical_refit_correlation <- vapply(
  refit_columns,
  function(columns) cor(corrected_covariates[[columns[1]]], corrected_covariates[[columns[2]]]),
  numeric(1)
)
covariate_summary$production_historical_refit_max_abs_difference <- vapply(
  refit_columns,
  function(columns) max(abs(corrected_covariates[[columns[1]]] - corrected_covariates[[columns[2]]])),
  numeric(1)
)
covariate_summary$historical_corrected_refit_correlation <- vapply(
  refit_columns,
  function(columns) cor(corrected_covariates[[columns[2]]], corrected_covariates[[columns[3]]]),
  numeric(1)
)
covariate_summary$historical_corrected_refit_max_abs_difference <- vapply(
  refit_columns,
  function(columns) max(abs(corrected_covariates[[columns[2]]] - corrected_covariates[[columns[3]]])),
  numeric(1)
)
sub144_row <- corrected_covariates[corrected_covariates$subjID == "sub-144", ]
write_tsv(covariate_summary, file.path(table_dir, "l3_covariate_correction_summary.tsv"))
write_tsv(
  data.frame(
    subjID = "sub-144",
    submitted_group_design_ev = sub144_row$submitted_group_rt,
    submitted_participant_table_RT = sub144_row$RT,
    submitted_event_mean_response_time = sub144_row$response_time_mean_submitted_events,
    corrected_event_mean_response_time = sub144_row$response_time_mean_corrected_events,
    mean_response_time_change = (
      sub144_row$response_time_mean_corrected_events -
        sub144_row$response_time_mean_submitted_events
    ),
    submitted_event_median_response_time = sub144_row$response_time_median_submitted_events,
    corrected_event_median_response_time = sub144_row$response_time_median_corrected_events,
    median_response_time_change = (
      sub144_row$response_time_median_corrected_events -
        sub144_row$response_time_median_submitted_events
    ),
    l3_policy = "retain submitted group RT pending exact transformation provenance"
  ),
  file.path(table_dir, "l3_group_rt_provenance.tsv")
)

fit_diagnostics <- function(model, event_source, model_quantity, partner) {
  messages <- model@optinfo$conv$lme4$messages
  data.frame(
    event_source = event_source,
    model_quantity = model_quantity,
    partner = partner,
    n_trials = nobs(model),
    singular = isSingular(model, tol = 1e-4),
    convergence_message = if (is.null(messages)) NA_character_ else paste(messages, collapse = " | "),
    log_likelihood = as.numeric(logLik(model))
  )
}
write_tsv(
  rbind(
    fit_diagnostics(historical_similar_fit, "submitted", "offer_slope", "similar"),
    fit_diagnostics(historical_dissimilar_fit, "submitted", "offer_slope", "dissimilar"),
    fit_diagnostics(similar_fit, "corrected", "offer_slope", "similar"),
    fit_diagnostics(dissimilar_fit, "corrected", "offer_slope", "dissimilar"),
    fit_diagnostics(historical_norm_similar_fit, "submitted", "offer3_random_intercept", "similar"),
    fit_diagnostics(historical_norm_dissimilar_fit, "submitted", "offer3_random_intercept", "dissimilar"),
    fit_diagnostics(norm_similar_fit, "corrected", "offer3_random_intercept", "similar"),
    fit_diagnostics(norm_dissimilar_fit, "corrected", "offer3_random_intercept", "dissimilar")
  ),
  file.path(table_dir, "l3_covariate_model_diagnostics.tsv")
)

# Unified model. A correlated interaction-slope model is singular; the explicit
# uncorrelated form is retained and the singular attempt is documented below.
unified_correlated <- glmer(
  accept ~ offer_c * similarity_num + (1 + offer_c * similarity_num | subjID),
  data = human,
  family = binomial,
  control = glmer_control
)
unified <- glmer(
  accept ~ offer_c * similarity_num + (1 + offer_c + similarity_num + offer_c:similarity_num || subjID),
  data = human,
  family = binomial,
  control = glmer_control
)
unified_table <- wald_table(unified, "unified_fairness_sensitivity", "revision-analysis", 47, nrow(human))
write_tsv(unified_table, file.path(table_dir, "unified_sensitivity_model.tsv"))

unified_random <- ranef(unified)$subjID
interaction_column <- grep("offer_c:similarity_num", names(unified_random), value = TRUE)
stopifnot(length(interaction_column) == 1)
sensitivity$unified_interaction_slope <- fixef(unified)["offer_c:similarity_num"] + unified_random[as.character(sensitivity$subjID), interaction_column]
sensitivity$tracked_minus_corrected <- sensitivity$submitted_metric_tracked - sensitivity$corrected_separate_model_metric
write_tsv(sensitivity, file.path(private_dir, "fairness_sensitivity_by_participant.tsv"))

sensitivity_diagnostics <- data.frame(
  metric = c(
    "tracked_vs_corrected_correlation",
    "tracked_vs_corrected_max_absolute_difference",
    "unified_vs_corrected_correlation",
    "correlated_unified_model_singular",
    "uncorrelated_unified_model_singular",
    "unified_random_interaction_sd"
  ),
  value = c(
    cor(sensitivity$submitted_metric_tracked, sensitivity$corrected_separate_model_metric),
    max(abs(sensitivity$tracked_minus_corrected)),
    cor(sensitivity$unified_interaction_slope, sensitivity$corrected_separate_model_metric),
    isSingular(unified_correlated, tol = 1e-4),
    isSingular(unified, tol = 1e-4),
    as.data.frame(VarCorr(unified))$sdcor[
      as.data.frame(VarCorr(unified))$var1 == "offer_c:similarity_num" & is.na(as.data.frame(VarCorr(unified))$var2)
    ]
  )
)
write_tsv(sensitivity_diagnostics, file.path(table_dir, "fairness_sensitivity_diagnostics.tsv"))

# Participant-level age comparison of the event-corrected version of the
# submitted separate-model sensitivity estimand.
sensitivity_age <- t.test(corrected_separate_model_metric ~ age_group, data = sensitivity)
younger_values <- sensitivity$corrected_separate_model_metric[sensitivity$age_group == "younger"]
older_values <- sensitivity$corrected_separate_model_metric[sensitivity$age_group == "older"]
n_y <- length(younger_values)
n_o <- length(older_values)
pooled_sd <- sqrt(((n_y - 1) * var(younger_values) + (n_o - 1) * var(older_values)) / (n_y + n_o - 2))
cohens_d <- (mean(younger_values) - mean(older_values)) / pooled_sd
hedges_correction <- 1 - 3 / (4 * (n_y + n_o) - 9)
hedges_g <- hedges_correction * cohens_d
hedges_g_se <- hedges_correction * sqrt((n_y + n_o) / (n_y * n_o) + cohens_d^2 / (2 * (n_y + n_o - 2)))
sensitivity_group_summary <- aggregate(
  corrected_separate_model_metric ~ age_group,
  data = sensitivity,
  FUN = function(x) c(n = length(x), mean = mean(x), sd = sd(x), min = min(x), max = max(x))
)
sensitivity_group_summary <- data.frame(
  age_group = sensitivity_group_summary$age_group,
  as.data.frame(sensitivity_group_summary$corrected_separate_model_metric),
  check.names = FALSE
)
write_tsv(sensitivity_group_summary, file.path(table_dir, "fairness_sensitivity_age_summary.tsv"))
write_tsv(
  data.frame(
    contrast = "younger-minus-older",
    estimate = unname(sensitivity_age$estimate[1] - sensitivity_age$estimate[2]),
    conf_low = sensitivity_age$conf.int[1],
    conf_high = sensitivity_age$conf.int[2],
    statistic = unname(sensitivity_age$statistic),
    degrees_freedom = unname(sensitivity_age$parameter),
    p_value = sensitivity_age$p.value,
    hedges_g = hedges_g,
    hedges_g_conf_low = hedges_g - 1.96 * hedges_g_se,
    hedges_g_conf_high = hedges_g + 1.96 * hedges_g_se
  ),
  file.path(table_dir, "fairness_sensitivity_age_test.tsv")
)

# Missed trials: participant is the analysis unit for the age comparison.
misses <- read.delim(file.path(private_dir, "missed_trials_by_participant.tsv"), check.names = FALSE)
misses$age_group <- factor(misses$age_group, levels = c("younger", "older"))
miss_test <- t.test(missed_total ~ age_group, data = misses)
miss_summary <- aggregate(
  missed_total ~ age_group,
  data = misses,
  FUN = function(x) c(n = length(x), total = sum(x), mean = mean(x), sd = sd(x), min = min(x), max = max(x))
)
miss_summary <- data.frame(
  age_group = miss_summary$age_group,
  as.data.frame(miss_summary$missed_total),
  check.names = FALSE
)
write_tsv(miss_summary, file.path(table_dir, "missed_trials_age_summary.tsv"))
write_tsv(
  data.frame(
    contrast = "younger-minus-older",
    estimate = unname(miss_test$estimate["mean in group younger"] - miss_test$estimate["mean in group older"]),
    conf_low = miss_test$conf.int[1],
    conf_high = miss_test$conf.int[2],
    statistic = unname(miss_test$statistic),
    degrees_freedom = unname(miss_test$parameter),
    p_value = miss_test$p.value
  ),
  file.path(table_dir, "missed_trials_age_test.tsv")
)

# Response time in the cleaned table is seconds after the response choices
# appeared (the cleaning step subtracts the task's fixed one-second delay).
rt_data <- subset(human, is.finite(response_time))
rt_model <- lmer(
  response_time ~ age_group * similarity + (1 + similarity | subjID),
  data = rt_data,
  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))
)
write_tsv(wald_table(rt_model, "response_selection_time", "revision-analysis", 47, nrow(rt_data)), file.path(table_dir, "response_time_model.tsv"))
rt_summary <- aggregate(response_time ~ age_group + similarity, data = rt_data, FUN = function(x) c(n = length(x), mean = mean(x), sd = sd(x)))
rt_summary <- data.frame(
  rt_summary[c("age_group", "similarity")],
  as.data.frame(rt_summary$response_time),
  check.names = FALSE
)
write_tsv(rt_summary, file.path(table_dir, "response_time_summary.tsv"))

# Population-level acceptance curves and observed source data.
prediction_grid <- expand.grid(
  offer = seq(min(human$offer), max(human$offer), by = 0.1),
  age_group = levels(human$age_group),
  similarity = levels(human$similarity),
  stringsAsFactors = FALSE
)
prediction_grid$age_group <- factor(prediction_grid$age_group, levels = levels(human$age_group))
prediction_grid$similarity <- factor(prediction_grid$similarity, levels = levels(human$similarity))
prediction_grid$offer_c <- prediction_grid$offer - mean(human$offer)
X <- model.matrix(delete.response(terms(primary_maximal)), prediction_grid)
beta <- fixef(primary_maximal)
V <- vcov(primary_maximal)
prediction_grid$linear_predictor <- as.vector(X %*% beta)
prediction_grid$linear_predictor_se <- sqrt(diag(X %*% V %*% t(X)))
prediction_grid$predicted_probability <- plogis(prediction_grid$linear_predictor)
prediction_grid$conf_low <- plogis(prediction_grid$linear_predictor - 1.96 * prediction_grid$linear_predictor_se)
prediction_grid$conf_high <- plogis(prediction_grid$linear_predictor + 1.96 * prediction_grid$linear_predictor_se)
write_tsv(prediction_grid, file.path(table_dir, "acceptance_curve_source_data.tsv"))

observed <- aggregate(accept ~ offer + age_group + similarity, data = human, FUN = function(x) c(n = length(x), accepted = sum(x), proportion = mean(x)))
observed <- data.frame(
  observed[c("offer", "age_group", "similarity")],
  as.data.frame(observed$accept),
  check.names = FALSE
)
write_tsv(observed, file.path(table_dir, "acceptance_observed_source_data.tsv"))

acceptance_plot <- ggplot(prediction_grid, aes(x = offer, y = predicted_probability, color = similarity, fill = similarity)) +
  geom_ribbon(aes(ymin = conf_low, ymax = conf_high), alpha = 0.14, color = NA) +
  geom_line(linewidth = 1) +
  geom_point(data = observed, aes(y = proportion), inherit.aes = TRUE, alpha = 0.45, size = 1.5) +
  facet_wrap(~age_group) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent_format()) +
  scale_color_manual(values = c(dissimilar = "#B55239", similar = "#28666E")) +
  scale_fill_manual(values = c(dissimilar = "#B55239", similar = "#28666E")) +
  labs(x = "Offer to participant ($)", y = "Acceptance probability", color = "Partner", fill = "Partner") +
  theme_classic(base_size = 11) +
  theme(legend.position = "top")
ggsave(file.path(figure_dir, "acceptance_curves.png"), acceptance_plot, width = 7.2, height = 4.2, dpi = 300)

sensitivity_plot <- ggplot(sensitivity, aes(x = age_group, y = corrected_separate_model_metric, color = age_group)) +
  geom_violin(aes(fill = age_group), alpha = 0.15, color = NA, width = 0.8) +
  geom_boxplot(width = 0.18, outlier.shape = NA, alpha = 0.25) +
  geom_jitter(position = position_jitter(width = 0.08, height = 0, seed = 20260907), size = 2, alpha = 0.75) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey40") +
  scale_color_manual(values = c(younger = "#345995", older = "#D1495B"), guide = "none") +
  scale_fill_manual(values = c(younger = "#345995", older = "#D1495B"), guide = "none") +
  labs(x = NULL, y = "Similar - dissimilar offer slope (log-odds)") +
  theme_classic(base_size = 11)
ggsave(file.path(figure_dir, "fairness_sensitivity_distribution.png"), sensitivity_plot, width = 4.8, height = 4.2, dpi = 300)

# Recover and deidentify the available explicit partner ratings.
partner_map <- c(`1` = "computer", `2` = "dissimilar", `3` = "similar")
trait_map <- c(`0` = "angry", `1` = "likeable", `2` = "fair", `4` = "satisfied")
rating_rows <- list()
for (participant in as.character(participants$subjID)) {
  numeric_id <- sub("^sub-", "", participant)
  for (session in 1:2) {
    path <- file.path(project_root, "source_data", "partner_ratings", numeric_id, sprintf("sub%s_Bargaining-Ratings-%d.csv", numeric_id, session))
    if (!file.exists(path)) next
    rating <- read.csv(path, stringsAsFactors = FALSE)
    partner_code <- suppressWarnings(as.integer(rating$Partner))
    trait_code <- suppressWarnings(as.integer(rating$Trait))
    rating_value <- suppressWarnings(as.numeric(rating$Rating))
    valid <- !is.na(partner_code) & !is.na(trait_code) & !is.na(rating_value)
    rating <- rating[valid, , drop = FALSE]
    partner_code <- partner_code[valid]
    trait_code <- trait_code[valid]
    rating_value <- rating_value[valid]
    rating_rows[[length(rating_rows) + 1]] <- data.frame(
      participant = participant,
      age_group = participants$age_group[match(participant, participants$subjID)],
      session = c(`1` = "pre", `2` = "post")[[as.character(session)]],
      record_index = seq_along(rating_value),
      partner = unname(partner_map[as.character(partner_code)]),
      trait = unname(trait_map[as.character(trait_code)]),
      rating = rating_value,
      stringsAsFactors = FALSE
    )
  }
}
ratings <- do.call(rbind, rating_rows)
ratings <- ratings[complete.cases(ratings[, c("partner", "trait", "rating")]), ]
write_tsv(ratings, file.path(private_dir, "partner_ratings_source_data.tsv"))

completeness <- expand.grid(participant = as.character(participants$subjID), session = c("pre", "post"), stringsAsFactors = FALSE)
record_counts <- aggregate(rating ~ participant + session, data = ratings, FUN = length)
names(record_counts)[3] <- "n_records"
completeness <- merge(completeness, record_counts, by = c("participant", "session"), all.x = TRUE, sort = FALSE)
completeness$n_records[is.na(completeness$n_records)] <- 0
completeness$expected_records <- ifelse(completeness$session == "pre", 6, 12)
completeness$status <- ifelse(
  completeness$n_records == 0,
  "missing",
  ifelse(completeness$n_records == completeness$expected_records, "complete", "duplicate_or_nonstandard")
)
write_tsv(completeness, file.path(private_dir, "partner_ratings_completeness.tsv"))

rating_tests <- list()
for (session_value in c("pre", "post")) {
  for (trait_value in sort(unique(ratings$trait[ratings$session == session_value]))) {
    subset_rows <- ratings[
      ratings$session == session_value & ratings$trait == trait_value & ratings$partner %in% c("similar", "dissimilar"),
    ]
    usable_participants <- completeness$participant[
      completeness$session == session_value & completeness$status == "complete"
    ]
    subset_rows <- subset_rows[subset_rows$participant %in% usable_participants, ]
    wide <- reshape(subset_rows[, c("participant", "partner", "rating")], idvar = "participant", timevar = "partner", direction = "wide")
    if (!all(c("rating.similar", "rating.dissimilar") %in% names(wide))) next
    wide <- wide[complete.cases(wide), ]
    test <- t.test(wide$rating.similar, wide$rating.dissimilar, paired = TRUE)
    rating_tests[[length(rating_tests) + 1]] <- data.frame(
      session = session_value,
      trait = trait_value,
      contrast = "similar-minus-dissimilar",
      n_pairs = nrow(wide),
      estimate = mean(wide$rating.similar - wide$rating.dissimilar),
      conf_low = test$conf.int[1],
      conf_high = test$conf.int[2],
      statistic = unname(test$statistic),
      degrees_freedom = unname(test$parameter),
      p_value = test$p.value
    )
  }
}
write_tsv(do.call(rbind, rating_tests), file.path(table_dir, "partner_ratings_human_contrasts.tsv"))

session_info <- sessionInfo()
software <- data.frame(
  component = c("R", "lme4", "lmerTest", "ggplot2"),
  version = c(
    as.character(getRversion()),
    as.character(packageVersion("lme4")),
    as.character(packageVersion("lmerTest")),
    as.character(packageVersion("ggplot2"))
  )
)
write_tsv(software, file.path(table_dir, "behavior_software_versions.tsv"))

cat(sprintf("PASS: reviewer behavior outputs written for %d participants and %d human trials\n", nlevels(human$subjID), nrow(human)))
