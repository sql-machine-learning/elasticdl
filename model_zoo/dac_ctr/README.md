# Convert Data to RecordIO Files

1. Download Kaggle Display Advertising Challenge Dataset by [link](https://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/)
2. Untar the "dac.tar.gz" and you will get a "train.txt"
3. Execute the command

```shell
python convert_to_recordio.py \
    --records_per_shard 400000 \
    --output_dir ./dac_records \
    --data_path xxx/dac/train.txt
```
