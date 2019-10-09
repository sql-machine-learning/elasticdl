# Release 0.1.0

## Major Features and Improvements
- This is the first release of ElasticDL. It supports TensorFlow 2.0.
- Includes a master-worker architecture, where the master controls task generation and entire job progress. Workers communicate with the master to get the tasks to execute and report execution results.
- Supports different job types: training-only, training-with-evaluation, evaluation-only and prediction-only.
- Provides high-level APIs and CLI for training, evaluation and prediction.
- Supports running in environments, including MiniKube, GCP, and on-prem clusters.
