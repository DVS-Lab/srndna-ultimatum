#!/usr/bin/env Rscript

# Parse tracked FSL group templates and audit their design/contrast structure.

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg), mustWork = TRUE)
root <- dirname(dirname(script_path))
output_dir <- file.path(root, "results", "reviewer", "tables")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

templates <- c(
  submitted_dmn_age = file.path(root, "templates", "L3_template_n47_ultimatum_twogroup_wCovs.fsf"),
  later_ecn_sensitivity_reconstruction = file.path(
    root,
    "legacy",
    "jen_working_tree",
    "templates",
    "L3_template_n47_ug_twogroup_wCovs_in-out_sensitivity2_logit.fsf"
  )
)

extract_one <- function(lines, pattern, replacement) {
  matches <- grep(pattern, lines, value = TRUE)
  if (length(matches) != 1) stop(sprintf("expected one match for %s, found %d", pattern, length(matches)))
  sub(pattern, replacement, matches)
}

parse_template <- function(label, path) {
  lines <- readLines(path, warn = FALSE)
  npts <- as.integer(extract_one(lines, "^set fmri\\(npts\\) ([0-9]+)$", "\\1"))
  nevs <- as.integer(extract_one(lines, "^set fmri\\(evs_orig\\) ([0-9]+)$", "\\1"))
  titles <- rep("", nevs)
  title_lines <- grep('^set fmri\\(evtitle[0-9]+\\) "', lines, value = TRUE)
  for (line in title_lines) {
    index <- as.integer(sub('^set fmri\\(evtitle([0-9]+)\\).*$', "\\1", line))
    titles[index] <- sub('^set fmri\\(evtitle[0-9]+\\) "(.*)"$', "\\1", line)
  }

  design <- matrix(NA_real_, nrow = npts, ncol = nevs)
  ev_lines <- grep("^set fmri\\(evg[0-9]+\\.[0-9]+\\) ", lines, value = TRUE)
  for (line in ev_lines) {
    row <- as.integer(sub("^set fmri\\(evg([0-9]+)\\..*$", "\\1", line))
    column <- as.integer(sub("^set fmri\\(evg[0-9]+\\.([0-9]+)\\).*$", "\\1", line))
    value <- as.numeric(sub("^set fmri\\(evg[0-9]+\\.[0-9]+\\) ", "", line))
    if (row <= npts && column <= nevs) design[row, column] <- value
  }
  if (anyNA(design)) stop(sprintf("incomplete design in %s", path))
  colnames(design) <- titles

  scaled <- sweep(design, 2, sqrt(colSums(design^2)), "/")
  rank <- qr(design)$rank
  factors <- if (rank == nevs) diag(solve(crossprod(design))) * colSums(design^2) else rep(Inf, nevs)

  inputs <- grep("^set feat_files\\([0-9]+\\) ", lines, value = TRUE)
  input_paths <- sub('^set feat_files\\([0-9]+\\) "(.*)"$', "\\1", inputs)
  group_lines <- grep("^set fmri\\(groupmem\\.[0-9]+\\) ", lines, value = TRUE)
  group_values <- as.numeric(sub("^set fmri\\(groupmem\\.[0-9]+\\) ", "", group_lines))

  contrast_names <- grep('^set fmri\\(conname_real\\.[0-9]+\\) "', lines, value = TRUE)
  contrast_rows <- list()
  for (name_line in contrast_names) {
    contrast_index <- as.integer(sub('^set fmri\\(conname_real\\.([0-9]+)\\).*$', "\\1", name_line))
    contrast_name <- sub('^set fmri\\(conname_real\\.[0-9]+\\) "(.*)"$', "\\1", name_line)
    values <- numeric(nevs)
    for (ev in seq_len(nevs)) {
      pattern <- sprintf("^set fmri\\(con_real%d\\.%d\\) (.*)$", contrast_index, ev)
      values[ev] <- as.numeric(extract_one(lines, pattern, "\\1"))
    }
    contrast_rows[[length(contrast_rows) + 1]] <- data.frame(
      template_id = label,
      contrast_index = contrast_index,
      contrast_name = contrast_name,
      contrast_vector = paste(format(values, trim = TRUE, scientific = FALSE), collapse = " ")
    )
  }

  list(
    summary = data.frame(
      template_id = label,
      template_path = sub(paste0("^", root, "/"), "", path),
      n_points = npts,
      n_evs = nevs,
      matrix_rank = rank,
      scaled_condition_number = kappa(scaled, exact = TRUE),
      max_design_variance_factor = max(factors),
      input_count = length(inputs),
      duplicate_input_count = length(input_paths) - length(unique(input_paths)),
      group_membership_values = paste(sort(unique(group_values)), collapse = ",")
    ),
    evs = data.frame(
      template_id = label,
      ev_index = seq_len(nevs),
      ev_title = titles,
      minimum = apply(design, 2, min),
      maximum = apply(design, 2, max),
      mean = colMeans(design),
      sd = apply(design, 2, sd),
      design_variance_factor = factors
    ),
    correlations = do.call(rbind, lapply(combn(seq_len(nevs), 2, simplify = FALSE), function(pair) {
      data.frame(
        template_id = label,
        ev_1 = titles[pair[1]],
        ev_2 = titles[pair[2]],
        correlation = cor(design[, pair[1]], design[, pair[2]])
      )
    })),
    contrasts = do.call(rbind, contrast_rows)
  )
}

audits <- Map(parse_template, names(templates), templates)
write.table(do.call(rbind, lapply(audits, `[[`, "summary")), file.path(output_dir, "l3_design_summary.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(do.call(rbind, lapply(audits, `[[`, "evs")), file.path(output_dir, "l3_design_ev_diagnostics.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(do.call(rbind, lapply(audits, `[[`, "correlations")), file.path(output_dir, "l3_ev_correlations.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(do.call(rbind, lapply(audits, `[[`, "contrasts")), file.path(output_dir, "l3_contrasts.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)

if (any(vapply(audits, function(x) x$summary$matrix_rank != x$summary$n_evs, logical(1)))) stop("rank-deficient tracked L3 design")
cat(sprintf("PASS: audited %d tracked L3 templates\n", length(audits)))
