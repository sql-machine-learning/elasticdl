To run this example in a Docker container, we need to build an image including PyTorch and matplotlib:

```bash
docker build -t swamp .
```

Then, we can run the example

```bash
docker run --rm -it -v $PWD:/work -w /work swamp python mnist.py \
    --trainer-number 2 \
    --loss-file loss.png \
    --pull-probability 0.5 \
    --loss-sample-interval 10 \
    --job-root-dir jobs \
    --delete-job-data True \
    --plot-validate-batch-size 64 \
    --plot-validate-max-batch 10
```

`mnist.py` writes an image `./loss.png` showing the loss curves of the parameter server and all trainers and the meaning of parameters in the above command are described below:

`--trainer-number` : number of trainers running in total.

`--loss-file` : output loss curve image file.

`--pull-probability` : the probability of trainer pulling from ps.

`--loss-sample-interval` : how many batches to wait before record a loss value. 

`--job-root-dir` : the storage path of job data(net and params). 

`--delete-job-data` : whether to delete job data after job finish and completes plot.

`--plot-validate-batch-size` : bach size for calculating accuracy corresponding validation dataset in plot stage.

`--plot-validate-max-batch` : max batch for calculating accuracy corresponding validation dataset in plot stage. 

An example with 2 trainer threads with the trainer pulling probability of 0, 0.5 and 1.0 respectively looks like the following:

![](curves/loss_with_pull_prob_0.png)

From the figure above, with the pulling probability 0, the effect of ps is simply observing the best trainer and do nothing.

![](curves/loss_with_pull_prob_0_5.png)

as the probability increases to 0.5 and 1.0, trainer is trapped in local optimum caused by a bad loss value of the last minibatch, loss experienced repeated shocks. and with probability of 0.5, loss in ps stopped updating very quickly, which possibly because pushed loss from tainer failed to pass double check of validation dataset.

![](curves/loss_with_pull_prob_1.png)

Note: to run the example on macOS, please remember to enlarge the amount of memory to the virtual machine that runs the Docker daemon.

![](docker-macos.png)
