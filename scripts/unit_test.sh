# Run Python unittests
pytest elasticdl/python/tests elasticdl_preprocessing/tests --cov=elasticdl/python --cov-report=xml
mkdir -p ./build
mv coverage.xml ./build