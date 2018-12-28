#!/bin/bash
total_trainer=$1
pull_probability_step=$2
for (( t = 1; t <= ${total_trainer}; t = t * 2 )); do
    for (( n = 0; n <= 10; n = n + ${pull_probability_step} )); do
        p=`awk 'BEGIN{printf "%.2f\n",('$n'/'10')}'`
	if [[ ! -f $f ]]; then
	    cmd="python train.py --model-sample-interval 10 --trainer-number $t --pull-probability $p"
	    echo Running $cmd
	    eval $cmd
	fi
    done
done

# re-compute the loss and accuracy.
python eval.py

# plot metrics curve graph.
python plot.py

# merge all the curve graphs into pdf.
python pdf_creator.py 
