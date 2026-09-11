#!/usr/bin/env Rscript

# Exploratory rating-by-choice analyses for the corrected Ultimatum data.
# Pre-task ratings are tested as predictors of choice. Post-task ratings are
# retained as descriptive consequences of the task and are not entered as
# prospective choice predictors.

suppressPackageStartupMessages({
  library(lme4)
  library(ggplot2)
})

options(contrasts = c("contr.treatment", "contr.poly"))

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
project_root <- dirname(dirname(script_path))
args <- commandArgs(trailingOnly = TRUE)

argument <- function(name, default = NULL, required = FALSE) {
  prefix <- paste0("--", name, "=")
  value <- args[startsWith(args, prefix)]
  if (length(value) == 1) return(sub(prefix, "", value, fixed = TRUE))
  if (required) stop(sprintf("missing required argument --%s=PATH", name))
  default
}

resolve_path <- function(path) {
  if (!grepl("^/", path)) path <- file.path(project_root, path)
  normalizePath(path, mustWork = TRUE)
}

ratings_path <- resolve_path(argument("ratings", required = TRUE))
trials_path <- resolve_path(argument("trials", required = TRUE))
sample_path <- resolve_path(argument("sample", "behavioral_analyses/data/participant_L3_47.csv"))
sensitivity_path <- resolve_path(argument("sensitivity", "results/reviewer/private/fairness_sensitivity_by_participant.tsv"))
association_path <- resolve_path(argument("associations", "results/reviewer/ultimatum_ratings/ultimatum_ratings_fairness_sensitivity.tsv"))
output_dir <- argument("output-dir", "results/reviewer/ultimatum_ratings")
figure_dir <- argument("figure-dir", "results/reviewer/figures")
if (!grepl("^/", output_dir)) output_dir <- file.path(project_root, output_dir)
if (!grepl("^/", figure_dir)) figure_dir <- file.path(project_root, figure_dir)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)
private_dir <- file.path(project_root, "results", "reviewer", "private")
dir.create(private_dir, recursive = TRUE, showWarnings = FALSE)

write_tsv <- function(x, path) {
  write.table(x, path, sep = "\t", row.names = FALSE, quote = FALSE, na = "n/a")
}

read_csv_clean <- function(path) {
  read.csv(path, check.names = FALSE, fileEncoding = "UTF-8-BOM")
}

ratings <- read.delim(ratings_path, check.names = FALSE, stringsAsFactors = FALSE)
ratings <- ratings[ratings$task == "ultimatum", ]
ratings$source_block <- as.integer(ratings$source_block)
ratings$response <- as.numeric(ratings$response)
sample <- read_csv_clean(sample_path)
sample$age_group <- factor(
  ifelse(sample$younger == 1, "younger", "older"),
  levels = c("younger", "older")
)
ratings <- ratings[ratings$participant_id %in% sample$subjID, ]

ambiguous_sessions <- data.frame(
  participant_id = c("sub-105", "sub-134", "sub-144", "sub-144"),
  timepoint = c("post", "post", "pre", "post"),
  stringsAsFactors = FALSE
)

select_policy <- function(data, policy) {
  if (policy == "exclude_ambiguous_sessions") {
    key <- paste(data$participant_id, data$timepoint)
    excluded <- paste(ambiguous_sessions$participant_id, ambiguous_sessions$timepoint)
    return(data[!key %in% excluded, ])
  }
  operation <- if (policy == "last_complete_block") max else min
  selected_blocks <- aggregate(
    source_block ~ participant_id + timepoint,
    data = data,
    FUN = operation
  )
  merge(
    data,
    selected_blocks,
    by = c("participant_id", "timepoint", "source_block"),
    all = FALSE,
    sort = FALSE
  )
}

rating_features <- function(selected, policy) {
  selected <- selected[
    selected$partner %in% c("similar", "dissimilar") &
      selected$rating_dimension %in% c("fairness", "likeability", "anger", "satisfaction"),
  ]
  keys <- c("participant_id", "timepoint", "rating_dimension", "partner")
  if (anyDuplicated(selected[keys])) stop(sprintf("policy %s leaves duplicate rating cells", policy))
  wide <- reshape(
    selected[, c(keys, "response")],
    idvar = c("participant_id", "timepoint", "rating_dimension"),
    timevar = "partner",
    direction = "wide"
  )
  if (!all(c("response.similar", "response.dissimilar") %in% names(wide))) {
    stop("ratings do not contain both human partners")
  }
  wide$rating_contrast <- wide$response.similar - wide$response.dissimilar
  wide$policy <- policy
  wide
}

policies <- c(
  "last_complete_block",
  "first_complete_block",
  "exclude_ambiguous_sessions"
)
features <- do.call(
  rbind,
  lapply(policies, function(policy) rating_features(select_policy(ratings, policy), policy))
)

trials <- read_csv_clean(trials_path)
trials <- merge(trials, sample[, c("subjID", "age_group")], by = "subjID")
human <- trials[trials$missed == 0 & trials$human == 1, ]
human$accept <- as.integer(as.character(human$accept))
human$similarity <- factor(
  ifelse(human$ingroup == 1, "similar", "dissimilar"),
  levels = c("dissimilar", "similar")
)
human$offer_c <- human$offer - mean(human$offer)
human$subjID <- factor(human$subjID)
human$age_group <- factor(human$age_group, levels = c("younger", "older"))
stopifnot(nlevels(human$subjID) == 47, nrow(human) == 4438)

glmer_control <- glmerControl(
  optimizer = "bobyqa",
  optCtrl = list(maxfun = 200000)
)

fit_model <- function(data, formula, model_id, policy, specification, random_effects, focal_terms) {
  fit <- tryCatch(
    suppressWarnings(glmer(formula, data = data, family = binomial, control = glmer_control)),
    error = function(error) error
  )
  if (inherits(fit, "error")) {
    diagnostics <- data.frame(
      model_id = model_id,
      policy = policy,
      specification = specification,
      random_effects = random_effects,
      n_participants = length(unique(data$subjID)),
      n_trials = nrow(data),
      singular = NA,
      convergence_message = conditionMessage(fit),
      log_likelihood = NA,
      AIC = NA
    )
    return(list(focal = NULL, full = NULL, diagnostics = diagnostics))
  }
  coefficient_table <- as.data.frame(coef(summary(fit)))
  coefficient_table$term <- rownames(coefficient_table)
  rownames(coefficient_table) <- NULL
  coefficient_table$model_id <- model_id
  coefficient_table$policy <- policy
  coefficient_table$specification <- specification
  coefficient_table$random_effects <- random_effects
  coefficient_table$n_participants <- length(unique(data$subjID))
  coefficient_table$n_trials <- nrow(data)
  names(coefficient_table)[1:4] <- c("estimate", "std_error", "statistic", "p_value")
  coefficient_table$conf_low <- coefficient_table$estimate - 1.96 * coefficient_table$std_error
  coefficient_table$conf_high <- coefficient_table$estimate + 1.96 * coefficient_table$std_error
  coefficient_table$rating_dimension <- NA_character_
  for (dimension in names(focal_terms)) {
    coefficient_table$rating_dimension[coefficient_table$term == focal_terms[[dimension]]] <- dimension
  }
  messages <- fit@optinfo$conv$lme4$messages
  diagnostics <- data.frame(
    model_id = model_id,
    policy = policy,
    specification = specification,
    random_effects = random_effects,
    n_participants = length(unique(data$subjID)),
    n_trials = nrow(data),
    singular = isSingular(fit, tol = 1e-4),
    convergence_message = if (is.null(messages)) NA_character_ else paste(messages, collapse = " | "),
    log_likelihood = as.numeric(logLik(fit)),
    AIC = AIC(fit)
  )
  list(
    focal = coefficient_table[!is.na(coefficient_table$rating_dimension), ],
    full = coefficient_table,
    diagnostics = diagnostics
  )
}

focal_rows <- list()
full_rows <- list()
diagnostic_rows <- list()
for (policy in policies) {
  pre <- features[features$policy == policy & features$timepoint == "pre", ]
  pre <- pre[pre$rating_dimension %in% c("fairness", "likeability"), ]
  pre_wide <- reshape(
    pre[, c("participant_id", "rating_dimension", "rating_contrast")],
    idvar = "participant_id",
    timevar = "rating_dimension",
    direction = "wide"
  )
  names(pre_wide) <- sub("^rating_contrast\\.", "", names(pre_wide))
  model_data <- merge(human, pre_wide, by.x = "subjID", by.y = "participant_id")
  model_data$fairness_z <- as.numeric(scale(model_data$fairness))
  model_data$likeability_z <- as.numeric(scale(model_data$likeability))
  model_data$subjID <- factor(model_data$subjID)

  for (dimension in c("fairness", "likeability")) {
    model_data$rating_z <- model_data[[paste0(dimension, "_z")]]
    for (random_effects in c("random_offer_slope", "random_intercept")) {
      random_term <- if (random_effects == "random_offer_slope") {
        "(1 + offer_c | subjID)"
      } else {
        "(1 | subjID)"
      }
      formula <- as.formula(paste(
        "accept ~ offer_c * age_group * similarity +",
        "offer_c * similarity * rating_z +", random_term
      ))
      model_id <- paste("separate", dimension, policy, random_effects, sep = "_")
      result <- fit_model(
        model_data,
        formula,
        model_id,
        policy,
        "separate_rating_model",
        random_effects,
        setNames("offer_c:similaritysimilar:rating_z", dimension)
      )
      if (!is.null(result$focal)) result$focal$focal_effect <- "common_rating_moderation"
      focal_rows[[length(focal_rows) + 1]] <- result$focal
      full_rows[[length(full_rows) + 1]] <- result$full
      diagnostic_rows[[length(diagnostic_rows) + 1]] <- result$diagnostics
    }
  }

  for (random_effects in c("random_offer_slope", "random_intercept")) {
    random_term <- if (random_effects == "random_offer_slope") {
      "(1 + offer_c | subjID)"
    } else {
      "(1 | subjID)"
    }
    formula <- as.formula(paste(
      "accept ~ offer_c * age_group * similarity +",
      "offer_c * similarity * fairness_z +",
      "offer_c * similarity * likeability_z +", random_term
    ))
    model_id <- paste("joint", policy, random_effects, sep = "_")
    result <- fit_model(
      model_data,
      formula,
      model_id,
      policy,
      "joint_rating_model",
      random_effects,
      c(
        fairness = "offer_c:similaritysimilar:fairness_z",
        likeability = "offer_c:similaritysimilar:likeability_z"
      )
    )
    if (!is.null(result$focal)) result$focal$focal_effect <- "common_rating_moderation"
    focal_rows[[length(focal_rows) + 1]] <- result$focal
    full_rows[[length(full_rows) + 1]] <- result$full
    diagnostic_rows[[length(diagnostic_rows) + 1]] <- result$diagnostics
  }

  # Secondary check prompted by the age-stratified descriptive scatterplot.
  # This tests whether rating moderation of the partner-specific offer slope
  # differs between older and younger adults in the trial-level model.
  for (dimension in c("fairness", "likeability")) {
    model_data$rating_z <- model_data[[paste0(dimension, "_z")]]
    formula <- accept ~ offer_c * age_group * similarity * rating_z +
      (1 + offer_c | subjID)
    model_id <- paste("age_interaction", dimension, policy, "random_offer_slope", sep = "_")
    result <- fit_model(
      model_data,
      formula,
      model_id,
      policy,
      "age_interaction_model",
      "random_offer_slope",
      setNames("offer_c:age_groupolder:similaritysimilar:rating_z", dimension)
    )
    if (!is.null(result$focal)) result$focal$focal_effect <- "age_difference_in_rating_moderation"
    focal_rows[[length(focal_rows) + 1]] <- result$focal
    full_rows[[length(full_rows) + 1]] <- result$full
    diagnostic_rows[[length(diagnostic_rows) + 1]] <- result$diagnostics
  }
}

focal <- do.call(rbind, focal_rows)
focal <- focal[, c(
  "model_id", "policy", "specification", "random_effects",
  "focal_effect", "rating_dimension", "term", "estimate", "std_error", "statistic",
  "p_value", "conf_low", "conf_high", "n_participants", "n_trials"
)]
diagnostics <- do.call(rbind, diagnostic_rows)
write_tsv(focal, file.path(output_dir, "rating_choice_moderation_focal_tests.tsv"))
write_tsv(diagnostics, file.path(output_dir, "rating_choice_moderation_model_diagnostics.tsv"))
write_tsv(do.call(rbind, full_rows), file.path(private_dir, "rating_choice_moderation_full_coefficients.tsv"))
write_tsv(features, file.path(private_dir, "ultimatum_rating_features_by_policy.tsv"))

# Figure 1: participant-level associations with the corrected behavioral score.
sensitivity <- read.delim(sensitivity_path, check.names = FALSE)
last_features <- features[features$policy == "last_complete_block", ]
scatter_features <- rbind(
  transform(
    last_features[last_features$timepoint == "pre" & last_features$rating_dimension == "likeability", ],
    panel = "Pre-task likeability"
  ),
  transform(
    last_features[last_features$timepoint == "post" & last_features$rating_dimension == "anger", ],
    panel = "Post-task anger"
  )
)
scatter_features <- merge(
  scatter_features,
  sensitivity[, c("subjID", "corrected_separate_model_metric")],
  by.x = "participant_id",
  by.y = "subjID"
)
scatter_features <- merge(
  scatter_features,
  sample[, c("subjID", "age_group")],
  by.x = "participant_id",
  by.y = "subjID"
)
associations <- read.delim(association_path, check.names = FALSE)
associations <- associations[
  associations$policy == "last_complete_block" &
    ((associations$rating_dimension == "likeability" & associations$rating_timepoint == "pre") |
      (associations$rating_dimension == "anger" & associations$rating_timepoint == "post")),
]
associations$panel <- ifelse(
  associations$rating_dimension == "likeability",
  "Pre-task likeability",
  "Post-task anger"
)
scatter_features$panel <- factor(
  scatter_features$panel,
  levels = c("Pre-task likeability", "Post-task anger")
)
associations$panel <- factor(
  associations$panel,
  levels = c("Pre-task likeability", "Post-task anger")
)
label <- data.frame(
  panel = associations$panel,
  label = sprintf(
    "Pearson r = %.2f, p = %.3f\nSpearman rho = %.2f, FDR q = %.3f",
    associations$pearson_r,
    associations$pearson_p,
    associations$spearman_rho,
    associations$spearman_p_fdr_bh
  ),
  stringsAsFactors = FALSE
)
scatter_plot <- ggplot(
  scatter_features,
  aes(x = rating_contrast, y = corrected_separate_model_metric, color = age_group)
) +
  geom_hline(yintercept = 0, linewidth = 0.35, linetype = "dashed", color = "grey65") +
  geom_vline(xintercept = 0, linewidth = 0.35, linetype = "dashed", color = "grey65") +
  geom_smooth(aes(group = 1), method = "lm", formula = y ~ x, se = TRUE, color = "grey25", fill = "grey75", linewidth = 0.8) +
  geom_point(size = 2.1, alpha = 0.82) +
  geom_text(
    data = label,
    aes(x = -Inf, y = Inf, label = label),
    inherit.aes = FALSE,
    hjust = -0.05,
    vjust = 1.15,
    size = 3.2,
    lineheight = 1.05
  ) +
  facet_wrap(~panel, scales = "free_x", nrow = 1) +
  scale_color_manual(values = c(younger = "#345995", older = "#D1495B")) +
  labs(
    x = "Similar - dissimilar rating difference (0-10 points)",
    y = "Behavioral fairness sensitivity\n(similar - dissimilar offer slope, log-odds)",
    color = "Age group"
  ) +
  theme_classic(base_size = 11) +
  theme(legend.position = "top", strip.background = element_blank(), strip.text = element_text(face = "bold"))
ggsave(
  file.path(figure_dir, "ratings_fairness_sensitivity_associations.png"),
  scatter_plot,
  width = 7.4,
  height = 4.0,
  dpi = 300
)

# Figure 2: focal rating moderation of the trial-level choice model.
plot_data <- focal[
  focal$random_effects == "random_offer_slope" &
    focal$focal_effect == "common_rating_moderation",
]
plot_data$policy_label <- factor(
  plot_data$policy,
  levels = policies,
  labels = c("Last block", "First block", "Exclude ambiguous sessions")
)
plot_data$specification_label <- factor(
  plot_data$specification,
  levels = c("separate_rating_model", "joint_rating_model"),
  labels = c("Separate", "Joint")
)
plot_data$rating_dimension <- factor(
  plot_data$rating_dimension,
  levels = c("likeability", "fairness"),
  labels = c("Pre-task likeability", "Pre-task fairness")
)
moderation_plot <- ggplot(
  plot_data,
  aes(
    x = estimate,
    y = rating_dimension,
    color = policy_label,
    shape = specification_label
  )
) +
  geom_vline(xintercept = 0, linewidth = 0.4, linetype = "dashed", color = "grey55") +
  geom_errorbar(
    aes(xmin = conf_low, xmax = conf_high),
    orientation = "y",
    width = 0,
    position = position_dodge(width = 0.48),
    linewidth = 0.7
  ) +
  geom_point(position = position_dodge(width = 0.48), size = 2.5) +
  scale_color_manual(values = c("#28666E", "#B55239", "#6C757D")) +
  labs(
    x = "Offer x similar partner x rating difference (log-odds)",
    y = NULL,
    color = "Rating rule",
    shape = "Model"
  ) +
  theme_classic(base_size = 11) +
  guides(
    color = guide_legend(nrow = 1),
    shape = guide_legend(nrow = 1)
  ) +
  theme(
    legend.position = "bottom",
    legend.box = "vertical",
    legend.margin = margin(t = 2, r = 2, b = 2, l = 2)
  )
ggsave(
  file.path(figure_dir, "ratings_choice_moderation.png"),
  moderation_plot,
  width = 7.4,
  height = 4.2,
  dpi = 300
)

cat(sprintf(
  "PASS: fitted %d rating-choice models for %d policies; wrote %d focal tests\n",
  nrow(diagnostics),
  length(policies),
  nrow(focal)
))
