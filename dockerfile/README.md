# Docker images

## devel

The image is intended to ease the developing process of ElasticDL.

- run `build_devel.sh` in MacOs or linux to build your own customized docker image. The build process adds your user and group on host machines to the image. This is to prevent the build process create artifacts with root permissions.

- run `dev.sh` to mount your host home directory as container home directory and you can build in the your git directories from there. Note that you might want to modify your `.bashrc` file to make it work on both host machine and in container.
