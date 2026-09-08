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
  write.table(x, path, sep = "\t", row.names = FALSE, quote = FALSE, na = "")
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
    convergence_message = if (is.null(messages)) "" else paste(messages, collapse = " | "),
    check.names = FALSE
  )
}

read_csv_clean <- function(path) read.csv(path, check.names = FALSE, fileEncoding = "UTF-8-BOM")

participants <- read_csv_clean(file.path(data_dir, "participant_L3_47.csv"))
names(participants)[1] <- "subjID"
participants$age_group <- factor(
  ifelse(participants$younger == 1, "younger", "older"),
  levels = c("younger", "older")
)

trials <- read_csv_clean(file.path(data_dir, "all_trials_brains.csv"))
trials <- merge(trials, participants[, c("subjID", "age_group")], by = "subjID")
human <- subset(trials, missed == 0 & human == 1)
human$accept <- as.integer(as.character(human$accept))
human$similarity <- factor(ifelse(human$ingroup == 1, "similar", "dissimilar"), levels = c("dissimilar", "similar"))
human$similarity_num <- as.integer(human$similarity == "similar")
human$offer_c <- human$offer - mean(human$offer)
human$subjID <- factor(human$subjID)
human$age_group <- factor(human$age_group, levels = c("younger", "older"))

stopifnot(nlevels(human$subjID) == 47, nrow(human) == 4439)

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
  wald_table(primary_maximal, "primary_maximal", "revision-refit-of-intended-model", 47, nrow(human)),
  wald_table(primary_intercept, "random_intercept_robustness", "revision-robustness", 47, nrow(human))
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

conditional_offer_slopes <- function(model, label) {
  values <- coef(model)$subjID$offer
  data.frame(subjID = rownames(coef(model)$subjID), value = values, stringsAsFactors = FALSE)
}
similar_slopes <- conditional_offer_slopes(similar_fit, "similar")
dissimilar_slopes <- conditional_offer_slopes(dissimilar_fit, "dissimilar")
names(similar_slopes)[2] <- "similar_offer_slope_rederived"
names(dissimilar_slopes)[2] <- "dissimilar_offer_slope_rederived"
rederived <- merge(similar_slopes, dissimilar_slopes, by = "subjID")
rederived$submitted_metric_rederived <- rederived$similar_offer_slope_rederived - rederived$dissimilar_offer_slope_rederived

submitted <- read_csv_clean(file.path(data_dir, "in_out_sensitivity_indiv_logit.csv"))
submitted <- submitted[, c("subjID", "in_out_indiv_logit")]
names(submitted)[2] <- "submitted_metric_tracked"
sensitivity <- merge(rederived, submitted, by = "subjID")
sensitivity <- merge(sensitivity, participants[, c("subjID", "age_group")], by = "subjID")

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
sensitivity$tracked_minus_rederived <- sensitivity$submitted_metric_tracked - sensitivity$submitted_metric_rederived
write_tsv(sensitivity, file.path(private_dir, "fairness_sensitivity_by_participant.tsv"))

sensitivity_diagnostics <- data.frame(
  metric = c(
    "tracked_vs_rederived_correlation",
    "tracked_vs_rederived_max_absolute_difference",
    "unified_vs_submitted_correlation",
    "correlated_unified_model_singular",
    "uncorrelated_unified_model_singular",
    "unified_random_interaction_sd"
  ),
  value = c(
    cor(sensitivity$submitted_metric_tracked, sensitivity$submitted_metric_rederived),
    max(abs(sensitivity$tracked_minus_rederived)),
    cor(sensitivity$unified_interaction_slope, sensitivity$submitted_metric_tracked),
    isSingular(unified_correlated, tol = 1e-4),
    isSingular(unified, tol = 1e-4),
    as.data.frame(VarCorr(unified))$sdcor[
      as.data.frame(VarCorr(unified))$var1 == "offer_c:similarity_num" & is.na(as.data.frame(VarCorr(unified))$var2)
    ]
  )
)
write_tsv(sensitivity_diagnostics, file.path(table_dir, "fairness_sensitivity_diagnostics.tsv"))

# Participant-level age comparison of the submitted sensitivity measure.
sensitivity_age <- t.test(submitted_metric_tracked ~ age_group, data = sensitivity)
younger_values <- sensitivity$submitted_metric_tracked[sensitivity$age_group == "younger"]
older_values <- sensitivity$submitted_metric_tracked[sensitivity$age_group == "older"]
n_y <- length(younger_values)
n_o <- length(older_values)
pooled_sd <- sqrt(((n_y - 1) * var(younger_values) + (n_o - 1) * var(older_values)) / (n_y + n_o - 2))
cohens_d <- (mean(younger_values) - mean(older_values)) / pooled_sd
hedges_correction <- 1 - 3 / (4 * (n_y + n_o) - 9)
hedges_g <- hedges_correction * cohens_d
hedges_g_se <- hedges_correction * sqrt((n_y + n_o) / (n_y * n_o) + cohens_d^2 / (2 * (n_y + n_o - 2)))
sensitivity_group_summary <- aggregate(
  submitted_metric_tracked ~ age_group,
  data = sensitivity,
  FUN = function(x) c(n = length(x), mean = mean(x), sd = sd(x), min = min(x), max = max(x))
)
sensitivity_group_summary <- data.frame(
  age_group = sensitivity_group_summary$age_group,
  as.data.frame(sensitivity_group_summary$submitted_metric_tracked),
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

sensitivity_plot <- ggplot(sensitivity, aes(x = age_group, y = submitted_metric_tracked, color = age_group)) +
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
