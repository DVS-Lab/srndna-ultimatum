#!/usr/bin/env bash

# ensure paths are correct irrespective from where user runs the script
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
maindir="$(dirname "$scriptdir")"
datadir=/data/projects/srndna-data

# study-specific inputs
sub=$1
run=$2
sm=6
TASK=ultimatum

# input/output paths
mkdir -p ${maindir}/derivatives/fsl/sub-${sub}_mc
outdir=${maindir}/derivatives/fsl/sub-${sub}_mc
summaryfile=${maindir}/derivatives/fsl/correlations-DVARSandMSSD.csv
mask=${maindir}/derivatives/fsl/sub-${sub}/L1_task-${TASK}_model-02_type-act_run-0${run}_sm-${sm}.feat/mask # may be better to use the fmriprep mask?
mcf=${datadir}/derivatives/fmriprep/sub-${sub}/func/sub-${sub}_task-${TASK}_run-${run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
mcfout=${outdir}/sub-${sub}_task-${TASK}_run-${run}_space-MNI152NLin2009cAsym_desc-preproc_bold

# define the number of volumes, and minus one to do the math
tmax=`fslval ${mcf} dim4`;
tmax1=`echo $tmax - 1 | bc`;

# set up the motion-corrected files
fslroi $mcf ${mcfout}1 0 $tmax1
fslroi $mcf ${mcfout}2 1 $tmax1

# generate MSSD (as per Nomi et al., 2017: https://www.jneurosci.org/content/jneuro/37/22/5539.full.pdf)
fslmaths ${mcfout}2 -sub ${mcfout}1 -mas ${mask} -sqr -Tmean ${outdir}/sub-${sub}_run-${run}_mssd3d -odt float

# for comparison DVARS (from fsl_motion_outliers): #
maskmean=`fslstats ${mask} -m`;
brainmed=`fslstats ${mcf} -k ${mask} -P 50`;
fslmaths ${mcfout}2 -sub ${mcfout}1 -mas ${mask} -sqr -Xmean -Ymean -Zmean -div $maskmean -sqrt ${outdir}/sub-${sub}_run-${run}_dvars4d -odt float
fslmaths ${outdir}/sub-${sub}_run-${run}_dvars4d -div $brainmed -mul 1000 ${outdir}/sub-${sub}_run-${run}_dvars4d
fslmaths ${outdir}/sub-${sub}_run-${run}_dvars4d -Tmean ${outdir}/sub-${sub}_run-${run}_dvarsTmean

# clean up intermediate files
rm -rf ${mcfout}1.nii.gz ${mcfout}2.nii.gz

