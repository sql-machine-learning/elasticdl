To convert the `report_cn.md` file into `report_cn.pdf`, please use the
following Dockerized tool.

```bash
(cd data; make) # Generate figures.
docker run --rm -v $PWD:/work cxwangyi/pandoc /mdtopdf.bash /work/report_cn.md
```
