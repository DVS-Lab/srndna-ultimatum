#!/usr/bin/env bash

# ensure paths are correct irrespective from where user runs the script
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
maindir="$(dirname "$scriptdir")"

# base paths
TASK=ultimatum
MAINOUTPUT=${maindir}/derivatives/fsl/L3_model-02_task-${TASK}_n47_flame1+2
outputdir=${maindir}/derivatives/imaging_plots
mkdir -p $outputdir

# activation: ROI name and other path information
for ROI in roi-VMPFC roi-VS roi-aINS roi-dACC func-DLPFC; do
	MASK=${maindir}/masks/${ROI}.nii.gz
	TYPE=act
	for COPENUM in 2 4 6; do # act
		cnum_padded=`zeropad ${COPENUM} 2`
		DATA=`ls -1 ${MAINOUTPUT}/L3_task-${TASK}_type-${TYPE}_cnum-${cnum_padded}_*_onegroup.gfeat/cope1.feat/filtered_func_data.nii.gz`
		fslmeants -i $DATA -o ${outputdir}/${ROI}_type-${TYPE}_cope-${cnum_padded}.txt -m ${MASK}
	done
done


# clust_img=$basedir/L3_task-${task}_${other}_cnum-${copenum}_cname-${copename}_${model}.gfeat/cope1.feat/cluster_mask_zstat${cov}.nii.gz
# MAX=`fslstats $i -R | awk '{ print $2 }'`
# Nclusters=`fslstats $clust_img -R | awk '{ print $2 }'`
# for c in `seq ${Nclusters}`; do
# 	fslmaths $clust_img -thr $c -uthr $c -bin ${NVdir}/cluster_${task}_${other}_${model}_${copename}_cov-${cov}_cluster${c}.nii.gz
# done


# connectivity: ROI name and other path information
for seedROI in "ecn func-insula" "dmn func-dACC"; do
	set -- $seedROI
	seed=$1
	ROI=$2
	TYPE=nppi-${seed}
	MASK=${maindir}/masks/${ROI}.nii.gz
	for COPENUM in 1 2 3 4 5 6; do
		cnum_padded=`zeropad ${COPENUM} 2`
		DATA=`ls -1 ${MAINOUTPUT}/L3_task-${TASK}_type-${TYPE}_cnum-${cnum_padded}_*_onegroup.gfeat/cope1.feat/filtered_func_data.nii.gz`
		fslmeants -i $DATA -o ${outputdir}/${ROI}_type-${TYPE}_cope-${cnum_padded}.txt -m ${MASK}
	done
done
