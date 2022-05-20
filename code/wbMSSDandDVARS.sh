#!/bin/bash

# This script will run a permutation test via randomise.
# Create the design.mat file with the Glm tool

# ensure paths are correct irrespective from where user runs the script
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
maindir="$(dirname "$scriptdir")"

# study-specific inputs and outputs
input=${maindir}/derivatives/fsl/mergedAllSubsRuns_mssd.nii.gz
output=${maindir}/derivatives/randomise_mssd_testing
design=${maindir}/derivatives/design_n100_testing.mat
contrasts=${maindir}/derivatives/design_n100_testing.con
mask=${maindir}/derivatives/fsl/mask.nii.gz

randomise -i $input -o $output -d $design -t $contrasts -m $mask -T -c 3.1 -n 10000
