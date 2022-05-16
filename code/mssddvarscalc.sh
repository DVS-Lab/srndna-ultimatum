#!/usr/bin/env bash

# ensure paths are correct irrespective from where user runs the script
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
maindir="$(dirname "$scriptdir")"
datadir=/data/projects/srndna-data
outdir=${maindir}/derivatives/fsl/sub-{$sub}

# study-specific inputs
sub=$1
run=$2
sm=6
TASK=ultimatum
mask=${outdir}/L1_task-${TASK}_model-02_type-act_run-0${run}_sm-${sm}.feat/mask
mcf=${datadir}/derivatives/fmriprep/sub-${sub}/func/sub-${sub}_task-${TASK}_run-${run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz


# generate MSSD (as per Nomi et al., 2017: https://www.jneurosci.org/content/jneuro/37/22/5539.full.pdf)
# MSSD: #  
$FSLDIR/bin/fslmaths ${mcf}2 -sub ${mcf}1 -mas ${mask} -sqr -Tmean ${outdir}_mc/mssd_3d -odt float

# for comparison DVARS (from fsl_motion_outliers): # 
$FSLDIR/bin/fslmaths ${mcf}2 -sub ${mcf}1 -mas ${mask} -sqr -Xmean -Ymean -Zmean -div $maskmean $sqrtcom ${outdir}_mc/dvars_4d -odt float
