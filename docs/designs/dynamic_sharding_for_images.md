# Dynamic Sharding Support Reading Original Images

Now, users have to convert images into RecordIO for ElasticDL. In practice,
many users read the original images to train their models. So, we need to
support reading the original data from the storage. However, there are various
file formats for the original data. We need to design a common definition
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
training data firstly. We can get the size by
`len(os.listdir("images"))` for case 1 and reading the CSV file for case 2.
For simplicity, we also
can store the image names in a CSV file for the first way to store images.
So, the name of the shard is the CSV file. The start and end indices are
the line number of the CSV file. When the worker gets the shard, it can read
the images by the lines in the CSV file.

For case 3, it is difficult to get the size  because we don't know the
format of the description. So, users need to indicate the size of
training data. We can design a function in the model definition and
users implement the function to return the size.

```python
from pycocotools.coco import COCO

def get_training_size():
    coco = COCO("annotations/captions_val2014.json")
    return len(coco.anns.keys())
```

Then, users should define the `training_data` as the Python file with the
function.

```bash
--training_data="/test_data/coco/train/create_shard.py"
```

The `PythonCustomReader` in ElasticDL can load the function to
get the total size in the master.

```python
class PythonCustomReader(object):
     def __init__(self, records_per_task):
        """
        Args:
            kwargs should contains "records_per_task".
        """
        AbstractDataReader.__init__(self, **kwargs)
        self._filename = self._kwargs["filename"]
        self._records_per_task = self._kwargs["records_per_task"]
        self._get_size_fn = None

    def load_get_size_fn(self, fn_name="get_training_size"):
        module = load_module(self._filename)
        self._get_size_fn = module[fn_name]

    def get_size(self):
        if self._get_size_fn:
            return self._get_size_fn()

    def create_shards(self):
        shard_name_prefix = "shard_"
        size = self.get_size()
        shards = {}
        num_shards = size // self._records_per_task
        start_ind = 0
        for shard_id in range(num_shards):
            shards[shard_name_prefix + str(shard_id)] = (
                start_ind,
                self._records_per_task,
            )
            start_ind += self._records_per_task
        return shards
```

Then, the master will call the function to get the size and split the
training data into shards by the size. The shard message will only contains
the start and end index. Users need to read the image information according
to the index by themselves.

## The Worker Creates the Dataset using Shards

### APIs to Fetch Shards

```python
class DataShardService(object):
    def __init__(self, batch_size, master_client=None,):
        self._mc = master_client
        if not self._mc
            master_addr = os.getenv("MASTER_ADDR")
            worker_id = os.getenv("WORKER_ID")
            self._mc = MasterClient(
                build_channel(master_addr), worker_id
            )
        self._pending_tasks = []
        self.record_count = 0

    def fetch_shard(self):
        return shard

    def report_batch_done(self):
        if task_done:
            report_task()
```

### Create Dataset Using TensorFlow

```python
import tensorflow as tf

global data_shard_service = DataShardService()

class DynamicShardingHook(tf.train.SessionRunHook):
    def __init__(self, num_worker):
        self._max_steps = max_steps
        self._local_step = 0
        self._batch_size = 256

    def after_run(self, run_context, run_values):
        self._local_step += 1
        if self._local_step * self._batch_size > data_shard_service.record_count:
            data_shard_service.report_batch_done()

def get_dataset(shuffle=False):
    def _record_generator():
        while True:
            shard = data_shard_service.fetch_shard()
            if not shard:
                break
            records = read_records(shard.start, shard.end)
            if shuffle:
                np.random.shuffle(records)
            for record in records:
                yield record
    return tf.data.Dataset.from_generator(_record_generator()
```

### Create Dataset Using PyTorch

Here, we create the dataset using COCO dataset.

```python
import torch
import cv2
from pycocotools.coco import COCO

global data_shard_service = DataShardService()

coco = COCO("annotations/captions_val2014.json")
ids = list(coco.anns.keys())

def read_images(shard):
    images = []
    for index in range(shard.start, shard.end):
        ann_id = ids[index]
        caption = coco.anns[ann_id]['segmentation']
        img_id = coco.anns[ann_id]['image_id']
        path = coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(image_path)
        images.append(image, caption)
    return images


class ImageDataset(torch.utils.data.IterableDataset):

    def __init__(self, shuffle=False):
        self._shuffle = shuffle

    def __iter__(self):
        while True:
            shard = data_shard_service.fetch_shard()
            if shard:
                images = read_images(shard)
                if self._shuffle:
                    np.random.shuffle(images)
                for image in images:
                    yield image
            else:
                break

data_loader = DataLoader(dataset=dataset, batch_size=32)
```
