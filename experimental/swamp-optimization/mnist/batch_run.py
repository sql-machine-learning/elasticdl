#! /usr/bin/env python

"""
Script to batch run trainers with given set of parameters.
batch_run.py -t 1 2 4 8 -p 0 0.5 1
"""
import argparse
import os


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--trainer",
        required=True,
        type=int,
        nargs="+",
        help="list of trainer numbers",
    )
    parser.add_argument(
        "-p",
        "--pull_prob",
        required=True,
        type=float,
        nargs="+",
        help="list of pull probabilities",
    )

    args = parser.parse_args()
    trainers = sorted(set(args.trainer))
    if not trainers or trainers[0] < 0:
        raise ValueError("Invalid trainer parameters:" + str(trainers))

    pull_probs = sorted(set(args.pull_prob))
    if not pull_probs or pull_probs[0] < 0 or pull_probs[-1] > 1:
        raise ValueError("Invalid trainer parameters:" + str(pull_probs))

    # Run trainer with all combinations
    commands = [
        (
            "python train.py --model-sample-interval 10 --trainer-number %d "
            "--pull-probability %f"
        )
        % (t, p)
        for t in trainers
        for p in pull_probs
    ]
    commands.extend(
        [
            # re-compute the loss and accuracy.
            "python eval.py",
            # plot metrics curve graph.
            "python plot.py",
            # merge all the curve graphs into pdf.
            "python pdf_creator.py",
        ]
    )

    print("commands to run:\n%s" % "\n".join(commands))
    for command in commands:
        print("-" * 20)
        print("Running:", command)
        res = os.system(command)
        if res != 0:
            print(u"\u001b[31;1mFailed!\u001b[0m")
        else:
            print(u"\u001b[32;1mSuccess!\u001b[0m")


if __name__ == "__main__":
    _main()
