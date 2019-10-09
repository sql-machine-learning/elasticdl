# Release 0.1.0

## Major Features and Improvements
- This is the first release of ElasticDL. It supports Tensorflow 2.0.
- Includes a master-worker architecture, where the master controls task generation and entire job progress. Workers talks to the master to get the tasks to execute and report execution results.
- Supports different job models: training-only, training-with-evaluation, evaluation-only and prediction-only.
- Provides high-level APIs and CLI for train, evaluation and prediction.
- Supports running environments including MiniKube, GCP, and on-prem clusters.
