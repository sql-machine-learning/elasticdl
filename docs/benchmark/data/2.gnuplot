set terminal pdf
set output "2.pdf"

set yrange [0:24]
set xlabel "training time (seconds)"
set ylabel "# utilized CPU"
set key right bottom
set grid
set title "Two jobs with different priorities running together"
plot '2.data' using 1:2 with lines \
     title 'a high-priority NGINX job auto scaling w.r.t. traffic', \
     '2.data' using 1:3 with lines \
     title 'a low-priority ElasticDL job scaling accordingly', \
     '2.data' using 1:4 with lines \
     title 'overall cluster utilization'

set terminal png
set output "2.png"
replot
