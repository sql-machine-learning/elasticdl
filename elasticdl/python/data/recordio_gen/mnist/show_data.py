import recordio
import sys
import argparse
import numpy as np

from contextlib import closing
from elasticdl.python.elasticdl.common.model_helper import load_module


def main(argv):
    parser = argparse.ArgumentParser(
        description="Show some data from mnist recordio"
    )
    parser.add_argument("file", help="RecordIo file to read")
    parser.add_argument(
        "--start", default=0, type=int, help="Start record number"
    )
    parser.add_argument(
        "--n", default=20, type=int, help="How many records to show"
    )
    parser.add_argument(
        "--codec_file",
        default="elasticdl/python/data/codec/tf_example_codec.py",
        help="Codec file name",
    )
    args = parser.parse_args(argv)

    # Initilize codec
    codec_module = load_module(args.codec_file)
    decode_fn = codec_module.codec.decode

    with closing(recordio.Scanner(args.file, args.start, args.n)) as f:
        for i in range(args.start, args.start + args.n):
            rec = f.record()
            if rec is None:
                break
            example = decode_fn(rec)
            image_array = example.features.feature['image'].float_list.value
            image_numpy = np.asarray(image_array).reshape(28, 28)
            label = example.features.feature['label'].int64_list.value[0]
            print("-" * 10)
            print("record:", i)
            print(image_numpy)
            print(label)


if __name__ == "__main__":
    main(sys.argv[1:])
