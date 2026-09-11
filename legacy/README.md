# Legacy material

This directory quarantines compact historical material that may be useful for
provenance but is not part of the active, reproducible analysis workflow.
Nothing here is an execution entry point.

Legacy material is organized first by source working tree and then by its
original role. Each source directory must document its origin, authority, and
retention policy. Large data, FEAT directories, generated images, exploratory
model outputs, and unrelated RL/DDM projects do not belong in Git; retain
those in the lab's read-only cold archive with a checksum manifest instead.

Active code belongs in `code/`, historical production templates in
`templates/`, canonical resubmission templates in `templates/revision/`, and
small validated results in `results/`.
