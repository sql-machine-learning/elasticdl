all : 1.png 2.png 3-1.png 3-2.png

# We use convert from ImageMagick to generate PNG files from PDF files, because
# Gnuplot replot command doesn't work well with multiplot used in 1.gnuplot.
# Also, the ratio of Gnuplot-generated PNG and PDF files differ from each other;
# ImageMagick keeps the ratio consistent.
%.png : %.pdf
	convert $< $@

%.pdf : %.gnuplot
	gnuplot $<

1.pdf : 1c.data 1s.data 1.gnuplot
	gnuplot 1.gnuplot

2.pdf : 2.data 2.gnuplot
	gnuplot 2.gnuplot

3-1.pdf : 3-1.gnuplot 3-1.data
	gnuplot $<

3-2.pdf : 3-2.gnuplot 3-2.data
	gnuplot $<

# tail -n +2 skips over the first one line of a file.  The sed command replaces
# the CSV separator by whitespace, as required by Gnuplot.
%.data : %.csv
	tail -n +2 $<  | sed 's/,/ /g' > $@

3-1.data : 3-1.csv
	tail -n +2 $<  | head -n 14 | sed 's/,/ /g' > $@

3-2.data : 3-2.csv
	tail -n +2 $<  | head -n 8 | sed 's/,/ /g' > $@

clean:
	rm *.data *.pdf *.png
