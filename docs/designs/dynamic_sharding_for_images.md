# Dynamic Sharding Support Reading Original Images

Now, users have to convert images into RecordIO for ElasticDL. In practice,
many users read the original images to train their models. So, we need to
support read the original data from the storage. However, the format to
store the original data is various. We need to design a common definition
for different formats.

## Different Ways to Store Images and Annotations

1. All images are in the same folder.

    ```txt
    |-- images
        |--0001.png
        |--0002.png
        |--0003.png
        |--0004.png
        |--0005.png
    ```

    In the format, users may not need labels in the image compression.

2. Images with the same label are in the same folder such as ImageNet.

    ```txt
    |-- images
        |--0
            |--0001.png
            |--0002.png
            |--0003.png
        |--1
            |--0004.png
            |--0005.png
            |--0006.png
    ```

    Besides the images, there is usually a file to store all filenames and labels
    like:

    ```csv
    0001.png,0
    0002.png,0
    0003.png,0
    0004.png,1
    0005.png,1
    0006.png,1
    ```

    Users will read the content from the file to read images from the storage.

3. The description of images, labels, and annotations is in a JSON or XML file.

    For example, the description of the image is in a JSON file for COCO
    dataset and in a XML file for Pascal VOC dataset. The example of COCO
    description is

    ```json
    "{'license': 3,
    'file_name': 'COCO_val2014_000000391895.jpg',
    'coco_url': 'http://images.cocodataset.org/val2014/COCO_val2014_000000391895.jpg',
    'height': 360,
    'width': 640,
    'date_captured': '2013-11-14 11:18:45',
    'flickr_url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg',
    'id': 391895}"

    ```

    ```json
    "{'image_id': 203564,
    'id': 37,
    'caption': 'A bicycle replica with a clock as the front wheel.'}"
    ```

    In elastic training, we need to split the training data into shards and
    assign those shards to workers. When the worker fails, we need to
    reassign uncompleted shards of the worker to other alive workers.

## Shard Definition

In ElasticDL, we define the shard information `start`, `end` and `shard_name`
in the task. We can define the shard independently and and expose the
shard for users to create their datasets.

```proto
message shard {
  // The storage name for data, such as a MaxCompute Table,
  // CSV file with image paths.
  string name = 1;

  // Starting and ending (non-inclusive) record number.
  int64 start = 2;
  int64 end = 3;
}
```

In order to split the training data into shards, we must get the size of the
training data firstly. For the former 2 ways, we can we can get the size by
`len(os.listdir("images"))` and reading the CSV file. For simplicity, we also
can store the image names in a CSV file for the first way to store images.
So, the name of the shard is the CSV file. The start and end indices are
the line number of the CSV file. When the worker gets the shard, it can read
the images by the lines in the CSV file.

It is difficult to get the size because we don't know the format of the
description. So, users need to indicate the size of training data. We can
design a function in the model definition and users implement the function
to return the size.

```python
def get_training_size():
    with open("annotations/captions_val2014.json") as f:
        data = json.load(f)
    return len(data["images"])
```

Then, the master will call the function to get the size and split the
training data into shards by the size. The shard message will only contains
the start and end index. Users need to read the image information according
to the index by themselves.

## The Worker Creates the Dataset using Shards

### APIs to Fetch Shards

```python
class DynamicShardingManager(object):
    def __init__(self):
        master_addr = os.getenv("MASTER_ADDR")
        worker_id = os.getenv("WORKER_ID")
        self.master_client = MasterClient(
            build_channel(master_addr), worker_id
        )
        self._pending_tasks = []
        self.record_count = 0

    def fetch_shard(self):
        shard = self.master_client.get_task().shard
        self.record_count += shard.end - shard.start
        self._pending_shards.append(shard)
        return shard

    def report_shard_done(self):
        task = self._pending_tasks.pop()
        self.master_client.report_task_result(task.id)
```

### Create Dataset Using TensorFlow

```python
import tensorflow as tf

global dynamic_sharding = DynamicShardingManager()

class DynamicShardingHook(tf.train.SessionRunHook):
    def __init__(self, num_worker, num_shards=100, max_steps=None):
        self._max_steps = max_steps
        self._local_step = 0
        self._batch_size = 256

    def after_run(self, run_context, run_values):
        self._local_step += 1
        if self._local_step * self._batch_size > dynamic_sharding.record_count:
            dynamic_sharding.report_shard_done()

def get_dataset():
    def _record_generator():
        while True:
            shard = dynamic_sharding.fetch_shard()
            records = read_records(shard.start, shard.end)
            for record in records:
                yield record
    return tf.data.Dataset.from_generator(_record_generator()
```

### Create Dataset Using PyTorch

```python
import torch

global dynamic_sharding = DynamicShardingManager()

class ImageDataset(torch.utils.data.IterableDataset):

    def __init__(self, shuffle=False):
        self._shuffle = shuffle

    def __iter__(self):
        while True:
            shard = dynamic_sharding.fetch_shard()
            if shard is not None:
                images = read_images(shard)
                if self._shuffle:
                    np.random.shuffle(images)
                for image in images:
                    yield image

data_loader = DataLoader(dataset=dataset, batch_size=32)
```
