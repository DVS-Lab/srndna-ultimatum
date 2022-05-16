#!/bin/bash

# ensure paths are correct irrespective from where user runs the script
scriptdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
basedir="$(dirname "$scriptdir")"

 for sub in 104 105 106 107 108 109 110 111 112 113 115 116 117 118 120 121 122 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 140 141 142 143 144 145 147 149 150 151 152 153 154 155 156 157 158 159; do
          nruns=2

          for run in `seq $nruns`; do
                # Manages the number of jobs and cores
                SCRIPTNAME=${basedir}/code/mssddvarscalc.sh
                NCORES=15
                while [ $(ps -ef | grep -v grep | grep $SCRIPTNAME | wc -l) -ge $NCORES ]; do
                        sleep 5s
                done
                bash $SCRIPTNAME $sub $run &
                sleep 1s
          done

 done
