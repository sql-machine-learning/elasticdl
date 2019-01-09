# Coordinator

Coordinator is a standalone server with following responsibilities:

* Provide the current best model to trainers
* Receive model and corresponding score from trainer
  * If the received score is better than current best score, double check using a batch of training data before accepting the model.
* periodically dump model to a directory
