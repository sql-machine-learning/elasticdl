# Docker images

The image is intended to ease the developing process of ElasticDL.

- Run `build_devel.sh` in macOS or Linux to build your own customized Docker image. The build process adds your user and group on host machine to the image. This is to prevent the build process from creating artifacts under the root permission.

- Run `dev.sh` to mount your host home directory as the container home directory so you can build in the your git directories from there. Note that you might want to modify your `.bashrc` file to make it work on both host machine and in container.
