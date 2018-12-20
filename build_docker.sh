#! /bin/bash
set -e
set -x

tmp_dir=$(mktemp -d)

# ignore temporary python files
cat > ${tmp_dir}/.dockerignore << EOF 
*.pyc
__pycache__
EOF

# copy ElasticDL code
cp -r python ${tmp_dir}
cp -r test ${tmp_dir}
cp launcher.py ${tmp_dir}

# build image
docker build -t elasticdl/user ${tmp_dir} -f- << EOF 
FROM reg.docker.alibaba-inc.com/elasticdl/base

ADD python /elasticdl
ADD test /elasticdl/test
ADD launcher.py /elasticdl

ENV PYTHONPATH /elasticdl/python
ENTRYPOINT ["python", "/elasticdl/launcher.py"]
EOF
