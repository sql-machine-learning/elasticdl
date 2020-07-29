## Benchmark of FTlib AllReduce

We perform experiments to test the performance of FTlib AllReduce in
ElasticDL on Minikube, the CPU cluster, and GPU cluster. We adopt
ResNet50 and MobileNetV2 in experiments. ResNet50 is a computation-intensive
model and MobileNetV2 is a communication-intensive model.

## Minikube

The setups of experiments are:

| Experiment parameter | Value    |
| -- | --- |
| Batch size | 64 |
| Batches per task | 50 |
| Images per task | 3220 |
| Dataset | cifar10 |
| Image shape | (32, 32, 3) |
| Worker resource| cpu=0.3,memory=2048Mi,ephemeral-storage=1024Mi|

### ResNet50

The number of trainable parameters in ResNet50 for cifar10 is 23,555,082.
The number of trainable tensors is 214.

| Workers |  computation/communication  |  Speed   |  Speedup Ratio |
| -- | --------------------------- | -------- | ------ |
| 1  |  -  | 3.1 images/s  |  1  |
| 2  | 10: 1 | 5.65 images/s | 1.82 |

### MobileNetV2

The number of trainable parameters in ResNet50 for cifar10 is 2,236,682.
The number of trainable tensors is 158.

| Workers   |  computation/communication  |  Speed   |  Speedup Ratio |
| -- | --------------------------- | -------- | ------ |
| 1  |   -  | 29 images/s  |  1  |
| 2  | 10: 3 | 44.7 images/s | 1.54 |
| 3  | 10: 6 | 57.2 images/s  | 1.97 |

## CPU Cluster

The setups of experiments are:

| Experiment parameter | Value |
| -- | --- |
| Batch size | 64 |
| Batches per task | 50 |
| Images per task | 3220 |
| Dataset | cifar10 |
| Image shape | (32, 32, 3) |
| Worker resource| cpu=4,memory=8192Mi,ephemeral-storage=1024Mi|

### ResNet50

| Workers   | communication   |  Speed   |  Speedup Ratio |
| -- | --------------------------- | -------- | ------ |
| 1  |   0%  | 26.7 images/s  |  1  |
| 2  | 18% | 41 images/s | 1.57 |
| 4  | 25% | 68.4 images/s  | 2.56 |
| 8  | 32% | 123 images/s | 4.61 |

### MobileNetV2

| Workers   | communication   |  Speed   |  Speedup Ratio |
| -- | --------------------------- | -------- | ------ |
| 1  |   0%  | 353.6 images/s  |  1  |
| 2  | 24% | 503 images/s | 1.42 |
| 4  | 44.7% | 680 images/s  | 1.92 |
| 8  | 66.7% | 648 images/s  | 1.83 |

### GPU

The setups of experiments are:

| Experiment parameter | Value    |
| -- | --- |
| Batch size | 64 |
| Batches per task | 16 |
| Images per task | 1024 |
| Dataset | Imagenet |
| Image shape | (256, 256, 3) |
| Worker resource| cpu=8,gpu=1,memory=16000Mi,ephemeral-storage=1024Mi|

### ResNet50

The number of trainable parameters in ResNet50 for cifar10 is 23,739,492.
The number of trainable tensors is 214.

| Workers   | speed | total task time  | allreduce time| tensor.numpy() time| apply_gradients |
| --------- | ----- | --------------- | -------- | ------ | ---------- |
| 1 (local) | 168 images/s| 6.1s  |  - | - |  4.16s |
| 2  | 148 images/s| 13.76s | 10.36s | 5.04s | 1.35s |
| 4  | 228 images/s| 18s | 14.67s |  5.14s | 1.30s |

![Resnet50](https://user-images.githubusercontent.com/18071380/88752262-0fdd2800-d18c-11ea-8b37-83ba5662e53d.png)

### MobileNetV2

The number of trainable parameters in ResNet50 for cifar10 is 2,386,084.
The number of trainable tensors is 158.

| Workers   | speed | total task time  | allreduce time| tensor.numpy() time| apply_gradients |
| --------- | ----- | --------------- | -------- | ------ | ---------- |
| 1 (local) | 169 images/s | 6.06s  |  - | - |  5.59s |
| 2  | 246 images/s | 8.34s  |   7.25026 | 5.79s | 0.6s |
| 4  | 401 images/s | 10.2029s | 8.9s |  5.78s | 0.71s |

![MobileNetV2](https://user-images.githubusercontent.com/18071380/88752339-43b84d80-d18c-11ea-9139-c907975a7aeb.png)

### An Image Compression Model with Conv2DTranspose

On GPU cluster, we also use an image compression model to test performance.
The model has less trainable tensors and more parameters than MobileNetV2.
The number of trainable parameters is 11,238,723 and the number of trainable
tensors is 34.

| Workers   | speed | total task time  | allreduce time| tensor.numpy() time| apply_gradients |
| --------- | ----- | --------------- | -------- | ------ | ---------- |
| 1 (local) | 109 images/s | 9.36s  |  - | - |  8.95s |
| 2  | 176 images/s | 11.65s | 1.47s | 9.36s | 0.42s |
| 4  | 328 images/s | 12.47s | 2.44s |  9.32s | 0.37s |

![Image Compression](https://user-images.githubusercontent.com/18071380/88752707-3059b200-d18d-11ea-84bd-1db670c64924.png)

For the above experiments of GPU, we can get summaries:

1. The speed-up ratio is better if the number of trainable weights is less.
2. The speed-up ratio is better if the computation is more complex.
3. It is weird that `apply_gradients` is so slow.
