# How to build

## Coordinator
Make sure you have necessary `grpc` python packages installed.
```bash
pip3 install grpcio grpcio-tools
```
In current directory run the following commands:
```bash
mkdir build
cd build
cmake ..
make coordinator
```

## Notes
Bazel's python grpc support is [not done](https://github.com/grpc/grpc/issues/8079) yet. We could use more advanced bazel `gen_rule` feature to run custom script but it might be difficult for novices, so we choose to use cmake instead.

There is another pitfall regarding generate python3 grpc code, after much digging I found this [comment](https://github.com/grpc/grpc/issues/9575#issuecomment-293934506), which solved the problem.
