all : 1.png 2.png 3-1.png 3-2.png

# We use convert from ImageMagick to generate PNG files from PDF files, because
# Gnuplot replot command doesn't work well with multiplot used in 1.gnuplot.
# Also, the ratio of Gnuplot-generated PNG and PDF files differ from each other;
# ImageMagick keeps the ratio consistent.
%.png : %.pdf
	convert $< $@

1.pdf : 1c.data 1s.data 1.gnuplot
	gnuplot 1.gnuplot

2.pdf : 2.data 2.gnuplot
	gnuplot 2.gnuplot

3-1.pdf : 3_wide_and_deep.data 3_wide_and_deep.gnuplot
	gnuplot 3_wide_and_deep.gnuplot

3-2.pdf : 3_xdeepfm.data 3_xdeepfm.gnuplot
	gnuplot 3_xdeepfm.gnuplot

# tail -n +2 skips over the first one line of a file.  The sed command replaces
# the CSV separator by whitespace, as required by Gnuplot.
%.data : %.csv
	tail -n +2 $<  | sed 's/,/ /g' > $@

clean:
	rm *.data *.pdf *.png
