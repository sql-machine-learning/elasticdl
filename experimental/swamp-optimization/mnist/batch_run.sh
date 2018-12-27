#!/bin/bash
for (( t = 1; t <= 64; t = t * 2 )); do
    for (( n = 0; n <= 10; n = n + 5 )); do
        p=`awk 'BEGIN{printf "%.2f\n",('$n'/'10')}'`
	if [[ ! -f $f ]]; then
	    cmd="python train.py --loss-sample-interval 10 --trainer-number $t --pull-probability $p"
	    echo Running $cmd
	    eval $cmd
	fi
    done
done
