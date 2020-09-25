import pandas as pd
from matplotlib import pyplot as plt

gang_gpu_df = pd.read_csv("../data/gang_allreduce_gpu.csv")
elastic_gpu_df = pd.read_csv("../data/elastic_allreduce_gpu.csv")

# fig = plt.figure(figsize=(15, 4))
fig = plt.figure(figsize=(13, 6))
ax = fig.add_subplot(211)
ax.plot(gang_gpu_df["time"], gang_gpu_df['gpu'])

plt.title(
    "Gang scheduling -- two jobs one after another",
    fontsize=14,
    fontfamily="Times New Roman"
)
plt.xlabel(xlabel='time(sec)', fontsize=14, fontfamily="Times New Roman")
plt.ylabel(ylabel='Utilized GPUs', fontsize=14, fontfamily="Times New Roman")
plt.xlim(0, 1000)
plt.ylim(0, 7)

# plt.subplot(2, 1, 2)
ax = fig.add_subplot(212)
ax.plot(elastic_gpu_df["time"], elastic_gpu_df["gpu"])
plt.title(
    "Elastic scheduling -- two jobs overlap and fully use the cluster",
    fontsize=14,
    fontfamily="Times New Roman"
)
plt.xlabel(xlabel='time(sec)', fontsize=14, fontfamily="Times New Roman")
plt.ylabel(ylabel='Utilized GPU', fontsize=14, fontfamily="Times New Roman")
plt.xlim(0, 1000)
plt.ylim(0, 7)
plt.subplots_adjust(hspace=0.5)
plt.show()
