#!/usr/bin/env Rscript

# Descriptive influence audit for the submitted DMN ROI plot. Because the ROI
# was selected from the group result, these diagnostics are not independent
# inferential tests and must not drive exclusions on their own.

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
root <- dirname(dirname(script_path))
table_dir <- file.path(root, "results", "reviewer", "tables")
private_dir <- file.path(root, "results", "reviewer", "private")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(private_dir, recursive = TRUE, showWarnings = FALSE)

p <- read.csv(file.path(root, "behavioral_analyses", "data", "participant_L3_47.csv"), fileEncoding = "UTF-8-BOM")
names(p)[1] <- "subjID"
p$older_group <- p$older
similar <- scan(file.path(root, "imaging_plots_SANS", "dmn_p_in-out_y-o_type-nppi-dmn_cope-04.txt"), quiet = TRUE)
dissimilar <- scan(file.path(root, "imaging_plots_SANS", "dmn_p_in-out_y-o_type-nppi-dmn_cope-06.txt"), quiet = TRUE)
stopifnot(nrow(p) == 47, length(similar) == nrow(p), length(dissimilar) == nrow(p))
p$roi_difference <- similar - dissimilar

model <- lm(roi_difference ~ older_group + ismale + tsnr + fd_mean + RT, data = p)
influence <- data.frame(
  subjID = p$subjID,
  age_group = ifelse(p$older == 1, "older", "younger"),
  roi_difference = p$roi_difference,
  cooks_distance = cooks.distance(model),
  studentized_residual = rstudent(model),
  leverage = hatvalues(model),
  dfbeta_age = dfbetas(model)[, "older_group"]
)
write.table(influence, file.path(private_dir, "dmn_roi_influence_by_participant.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

leave_one_out <- lapply(seq_len(nrow(p)), function(index) {
  fit <- lm(roi_difference ~ older_group + ismale + tsnr + fd_mean + RT, data = p[-index, ])
  table <- coef(summary(fit))
  c(estimate = table["older_group", "Estimate"], p_value = table["older_group", "Pr(>|t|)"])
})
leave_one_out <- do.call(rbind, leave_one_out)
age_row <- coef(summary(model))["older_group", ]
threshold <- 4 / nrow(p)
summary <- data.frame(
  analysis = "descriptive_selected_dmn_roi",
  n_participants = nrow(p),
  age_estimate = age_row["Estimate"],
  age_std_error = age_row["Std. Error"],
  age_t = age_row["t value"],
  age_p = age_row["Pr(>|t|)"],
  cooks_threshold_4_over_n = threshold,
  n_above_cooks_threshold = sum(influence$cooks_distance > threshold),
  max_cooks_distance = max(influence$cooks_distance),
  max_absolute_studentized_residual = max(abs(influence$studentized_residual)),
  max_leverage = max(influence$leverage),
  max_absolute_dfbeta_age = max(abs(influence$dfbeta_age)),
  leave_one_out_age_estimate_min = min(leave_one_out[, "estimate"]),
  leave_one_out_age_estimate_max = max(leave_one_out[, "estimate"]),
  leave_one_out_age_p_min = min(leave_one_out[, "p_value"]),
  leave_one_out_age_p_max = max(leave_one_out[, "p_value"]),
  interpretive_limit = "ROI selected from group result; descriptive only; image-level robustness required"
)
write.table(summary, file.path(table_dir, "dmn_roi_influence_summary.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
cat("PASS: wrote descriptive DMN ROI influence audit\n")
