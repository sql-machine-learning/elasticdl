# RecordIO

RecordIO is a file format created for [PaddlePaddle Elastic Deep Learning](https://kubernetes.io/blog/2017/12/paddle-paddle-fluid-elastic-learning/).  It is generally useful for distributed computing.

## Motivations

In distributed computing, we often need to dispatch tasks to worker processes.  Usually, a task is defined as a parition of the input data, like what MapReduce and distributed machine learning do.

Most distributed filesystems, including HDFS, Google FS, and CephFS, prefer a small number of big files.  Therefore, it is impratical to create each task as a small file; instead, we need a format for big files that is

1. appenable, so that applications can append records to the file without updating the meta-data, thus fault tolerable,
2. partitionable, so that applications can quickly scan over the file to count the total number of records, and create tasks each corresponds to a sequence of records.

RecordIO is such a file format.

## Write 

```python
data = open('demo.recordio', 'wb')
max_chunk_size = 1024
writer = Writer(data, max_chunk_size)
writer.write('abc')
writer.write('edf')
writer.flush()
data.close()
```

## Read

```python
data = open('demo.recordio', 'rb')   
index = FileIndex(data)
print('Total file records: ' + str(index.total_records()))

for i in range(index.total_chunks()):
  reader = Reader(data, index.chunk_offset(i))
  print('Total chunk records: ' + str(reader.total_count()))

  while reader.has_next():
    print('record value: ' + reader.next())

data.close()
```

## RecordIOFile (wrapper for easier use)
```python
# write
rdio_w = RecordIOFile('demo.recordio', 'w')
rdio_w.write('abc')
rdio_w.write('def')
rdio_w.close()

# read
rdio_r = RecordIOFile('demo.recordio', 'r')
iterator = rdio_r.iterator()       
while iterator.has_next():
    record = iterator.next()
rdio_r.close()
```
