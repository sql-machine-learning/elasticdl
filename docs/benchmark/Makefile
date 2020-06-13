all: report_cn.pdf

report_cn.pdf : report_cn.md data
	docker run --rm -v ${PWD}:/work cxwangyi/pandoc /mdtopdf.bash /work/$<

data:
	$(MAKE) -C $@
