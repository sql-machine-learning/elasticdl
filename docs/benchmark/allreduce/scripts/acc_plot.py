import pandas as pd
from matplotlib import pyplot as plt


if __name__ == "__main__":
    df = pd.read_csv("../data/cifar10_resnet20_acc.csv")

    f = plt.figure(figsize=(10, 8))

    x = df['epoch_index']
    y = df['baseline_4workers_0']
    plt.plot(x, y, '--', label="baseline_4workers")

    y = df['baseline_4workers_1']
    plt.plot(x, y, '--', label="baseline_4workers")

    y = df['baseline_2workers_0']
    plt.plot(x, y, '-.', label="baseline_2workers")

    y = df['baseline_2workers_1']
    plt.plot(x, y, '-.', label="baseline_2workers")

    y = df['elastic_2_4workers_0']
    plt.plot(x, y, label="elastic_2_4workers")

    y = df['elastic_2_4workers_1']
    plt.plot(x, y, label="elastic_2_4workers")

    plt.title("The Accuracy of Resnet20 on cifar10 test dataset")
    plt.xlabel(xlabel='Iteration epoches', fontsize=18, fontfamily="Times New Roman")
    plt.ylabel(ylabel='Accuracy', fontsize=18, fontfamily="Times New Roman")
    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98)) 
    plt.ylim((0.4, 1))
    plt.show()
