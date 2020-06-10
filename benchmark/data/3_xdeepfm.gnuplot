set terminal pdf
set output "3-2.pdf"

set xlabel "iterations"
set ylabel "AUC"
set key right bottom
set grid
set title "Changing # workers doesn't affect convergence -- xDeepFM model"
plot '3_xdeepfm.data' using 1:2 with lines lt rgb "violet" title 'gang 4 workers', \
     '3_xdeepfm.data' using 1:3 with lines lt rgb "violet" title 'gang 4 workers', \
     '3_xdeepfm.data' using 1:4 with lines lt rgb "violet" title 'gang 4 workers', \
     '3_xdeepfm.data' using 1:5 with lines lt rgb "blue" title 'gang 8 workers', \
     '3_xdeepfm.data' using 1:6 with lines lt rgb "blue" title 'gang 8 workers', \
     '3_xdeepfm.data' using 1:7 with lines lt rgb "blue" title 'gang 8 workers', \
     '3_xdeepfm.data' using 1:8 with lines lt rgb "green" title 'elastic 4\~8 workers', \
     '3_xdeepfm.data' using 1:9 with lines lt rgb "green" title 'elastic 4\~8 workers', \
     '3_xdeepfm.data' using 1:10 with lines lt rgb "green" title 'elastic 4\~8 workers'

set terminal png
set output "3-2.png"
replot
