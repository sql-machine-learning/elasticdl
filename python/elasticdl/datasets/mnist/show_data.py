from recordio.file import File
import record
import sys
import argparse


def main(argv):
    print(argv)
    parser = argparse.ArgumentParser(
        description="Show same data from mnist recordio"
    )
    parser.add_argument("file", help="RecordIo file to read")
    parser.add_argument(
        "--start", default=0, type=int, help="Start record number"
    )
    parser.add_argument("--step", default=1, type=int, help="Step")
    parser.add_argument(
        "--n", default=20, type=int, help="How many record to show"
    )
    args = parser.parse_args(argv)

    with File(args.file, "r") as f:
        for i in range(
            args.start, args.start + (args.n * args.step), args.step
        ):
            print("-" * 10)
            print("record:", i)
            record.show(*record.decode(f.get(i)))


if __name__ == "__main__":
    main(sys.argv[1:])
