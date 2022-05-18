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
outdir=${maindir}/derivatives/fsl/sub-${sub}
mask=${outdir}/L1_task-${TASK}_model-02_type-act_run-0${run}_sm-${sm}.feat/mask # may be better to use the fmriprep mask?
mcf=${datadir}/derivatives/fmriprep/sub-${sub}/func/sub-${sub}_task-${TASK}_run-${run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz

# define the number of volumes, and minus one to do the math
tmax=`fslval ${mcf} dim4`;
tmax1=`echo $tmax - 1 | bc`;

# set up the motion-corrected files
fslroi $mcf ${mcf}1 0 $tmax1
fslroi $mcf ${mcf}2 1 $tmax1

# generate MSSD (as per Nomi et al., 2017: https://www.jneurosci.org/content/jneuro/37/22/5539.full.pdf)
fslmaths ${mcf}2 -sub ${mcf}1 -mas ${mask} -sqr -Tmean ${outdir}/sub-${sub}_run-${run}_mssd3d -odt float

# for comparison DVARS (from fsl_motion_outliers): #
maskmean=`fslstats ${mask} -m`;
brainmed=`fslstats ${mcf} -k ${mask} -P 50`;
fslmaths ${mcf}2 -sub ${mcf}1 -mas ${mask} -sqr -Xmean -Ymean -Zmean -div $maskmean -sqrt ${outdir}/sub-${sub}_run-${run}_dvars4d -odt float
fslmaths ${outdir}/sub-${sub}_run-${run}_dvars4d -div $brainmed -mul 1000 ${outdir}/sub-${sub}_run-${run}_dvars4d
fslmaths sub-${sub}_run-${run}_dvars4d -Tmean sub-${sub}_run-${run}_dvarsTmean

# correlate MSSD and DVARS (will want to output this to a file)
fslcc -t -1 --noabs -m $mask sub-${sub}_run-${run}_dvars4d sub-${sub}_run-${run}_dvarsTmean ${outdir}/sub-${sub}_run-${run}_mssd3d

# clean up intermediate files
rm -rf ${mcf}1.nii.gz ${mcf}2.nii.gz
