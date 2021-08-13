#!/bin/bash

# This run_* script is a wrapper for L3stats.sh, so it will loop over several
# copes and models. Note that Contrast N for PPI is always PHYS in these models.


# ensure paths are correct irrespective from where user runs the script
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
maindir="$(dirname "$scriptdir")"


# this loop defines the different types of analyses that will go into the group comparisons
for analysis in act nppi-dmn nppi-ecn; do
	analysistype=type-${analysis}

	# these define the cope number (copenum) and cope name (copename)
	# "1 comp" "2 comp_p" "3 in" "4 in_p" "5 out" "6 out_p"
	# "7 in_p-out_p" "8 soc_p-nonsoc_p" "9 in-out" "10 soc-nonsoc" "11 phys"
	for copeinfo in "1 comp" "2 comp_p" "3 in" "4 in_p" "5 out" "6 out_p" "7 in_p-out_p" "8 soc_p-nonsoc_p" "9 in-out" "10 soc-nonsoc" "11 phys"; do

		# split copeinfo variable
		set -- $copeinfo
		copenum=$1
		copename=$2

		if [ "${analysistype}" == "type-act" ] && [ "${copeinfo}" == "11 phys" ]; then
			echo "skipping phys for activation since it does not exist..."
			continue
		fi


		NCORES=9
		SCRIPTNAME=${maindir}/code/L3stats.sh
		while [ $(ps -ef | grep -v grep | grep $SCRIPTNAME | wc -l) -ge $NCORES ]; do
			sleep 1s
		done
		bash $SCRIPTNAME $copenum $copename $analysistype &

	done
done
