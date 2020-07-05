set terminal pdf
set output "1.pdf"

set multiplot layout 2,1 rowsfirst
set bmargin 3

set yrange [0:24]
set xrange [0:800]
set xlabel "training time (seconds)"
set ylabel "# utilized CPU"
set grid
set title "Gang scheduling -- two jobs one after another"
plot '1c.data' with lines notitle

set yrange [0:24]
set xrange [0:800]
set xlabel "training time (seconds)"
set ylabel "# utilized CPU"
set grid
set title "Elastic scheduling -- two jobs overlap and fully use the cluster"
plot '1s.data' with lines notitle

unset multiplot

set terminal png
set output "1.png"
replot
