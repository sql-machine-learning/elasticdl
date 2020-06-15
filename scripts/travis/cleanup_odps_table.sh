docker run --rm -it \
    -e MAXCOMPUTE_TABLE=$MAXCOMPUTE_TABLE \
    -e MAXCOMPUTE_PROJECT=$MAXCOMPUTE_PROJECT \
    -e MAXCOMPUTE_AK=$ODPS_ACCESS_ID \
    -e MAXCOMPUTE_SK=$ODPS_ACCESS_KEY \
    -v $PWD:/work \
    -w /work \
    elasticdl:dev \
    bash -c 'python -c '`
    `'"from elasticdl.python.tests.test_utils import *; '`
    `'delete_iris_odps_table_from_env()"'