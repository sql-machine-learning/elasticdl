import argparse
import sys
from tensorflow.ps import ParameterServer

class ThreadLauncher(object):
    def launch(prog, num_ps, num_worker, input):
        # launch ps
        ps = []

def main(args):
    def pos_int(x):
        if x < 0: raise ValueError('Positive integer required')
    
    parser = argparse.ArgumentParser(description='ElasticDL launcher.')
    parser.add_argument('script', help='training script')
    parser.add_argument('--class_name', required=True, help='python class that holds the model definition and optimizer, etc.')
    parser.add_argument('--runner', required=True, help='training process runner')
    parser.add_argument('--num_ps', type=pos_int, required=True, help='number of parameter servers')
    parser.add_argument('--num_worker', type=pos_int, required=True, help='number of workers')
    parser.add_argument('--input', required=True, help='base path that contains the input data in RecordIo format')
    # TODO(l.zou): add support for passing arguments to the script.

    args = parser.parse_args(args)


if __name__ == '__main__':
    main(sys.argv)




