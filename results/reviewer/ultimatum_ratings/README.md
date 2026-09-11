# Exploratory Ultimatum ratings analysis

These tables provide a preliminary, descriptive analysis of the partner ratings
collected before and after the Ultimatum task. They are not confirmatory tests.
The analysis is limited to the 47-participant imaging sample in
`behavioral_analyses/data/participant_L3_47.csv`.

The source audit found changed repeated Ultimatum blocks for sub-105, sub-134,
and sub-144. The `last_complete_block` policy is the provisional primary
scenario, not yet a final curation decision. Every test is repeated using the
first complete block and after excluding the four ambiguous participant-session
files. This last scenario matches the conservative session-level handling used
by the earlier reviewer ratings table.

Run from the `srndna-ultimatum` repository root after cloning
`srndna-datapaper` beside it:

```bash
python3 code/analyze_ultimatum_ratings.py \
  --ratings ../srndna-datapaper/results/ratings_audit/ratings_normalized_rows.tsv \
  --sample behavioral_analyses/data/participant_L3_47.csv \
  --participants ../srndna-datapaper/bids/participants.tsv \
  --output-dir results/reviewer/ultimatum_ratings
```

Outputs:

- `ultimatum_ratings_sample.tsv`: rating availability and block-selection
  provenance for all 47 paper participants.
- `ultimatum_ratings_descriptives.tsv`: means, SDs, SEMs, and medians overall
  and by age group under all three resolution policies.
- `ultimatum_ratings_within_subject_tests.tsv`: paired partner contrasts,
  pre/post changes, effect sizes, and within-family FDR adjustments.
- `ultimatum_ratings_age_tests.tsv`: exploratory exact-age correlations and
  younger/older comparisons for the provisional primary policy.

Sub-143 has no ratings source file. Sub-144's ratings files are unique to that
participant and were committed with that participant's raw task logs; neither
file is a content duplicate of another participant's ratings. The earlier
sub-143/sub-144 Ultimatum event-file problem therefore does not propagate into
these rating responses. Any later analysis relating ratings to task behavior
must use the corrected event-derived behavioral table rather than the legacy
per-trial table.
