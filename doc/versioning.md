# Versioning

For better managing and releasing code, here is a summary about the versioning story of ElasticDL. Some previous discussions can be found [here](#1121).

## Branches

There are three different type branches:
- **develop** branch: the main branch where the source code always reflects a state with the latest delivered development changes for the next release.
- **feature** branches: branch off from the `develop` branch and merge back into `develop`. `Feature` branches are used to develop new features, and created/maintained by developers themselves. `Feature` branches will be deleted after code changes get merge info `develop` branch.
- **release** branches: branch off from the `develop` branch. When the `develop` branch reaches the desired state of a new release, cut off a new release branch, and start to publish new versions there. Release branches will be kept.

For version numbers, we follow the [Semantic Versioning 2.0.0](https://semver.org/ with style MAJOR.MINOR.PATCH (e.g. 0.2.1). For each MAJOR.MINOR, we keep a corresponding `release` branch (named `branch-x.y`). Each following patch version x.y.z is released from thie `branch-x.y`.


## Feature Development

New features and bugfixes are developed in `feature` branches. Make sure you work with the latest version of `develop` branch.

```bash
# Start a new feature branch from develop branch
$ git checkout -b feature1 develop
# Do some code changes locally
$ git commit -a -m "Add some feature description"
$ git push origin feature1
# Create a PR for feature1 in https://github.com/sql-machine-learning/elasticdl
# Once the PR gets accepted and merged, it is ok to delete the branch locally
$ git checkout develop
$ git branch -D feature1
```

## Releasing

Here are detailed steps for releasing an example version 0.1.0.

```bash
# Start a new release branch from develop branch
$ git checkout -b branch-0.1 develop
$ git push origin branch-0.1
# Prepare the first release candidate version 0.1.0rc0
$ ./bump_version.sh 0.1.0rc0
# Update RELEASE.md to include major changes in this version
# Then commit changes
$ git commit -a -m "Release 0.1.0rc0"
$ git push origin branch-0.1
# Publish 0.1.0rc0 to PyPI
```

So now we have 0.1.0rc0 ready to use. Test out this version. If any issues found, get them fixed in `develop` branch, merged into `branch-0.1` branch, and repeat the aforementioned steps to publish a new release candidate version. We keep releasing `rc` version until no further issues found. At that time, release the official `0.1.0` branch.

```bash
$ git checkout -b branch-0.1 develop
$ ./bump_version.sh 0.1.0
$ git commit -a -m "Release 0.1.0"
$ git push origin branch-0.1
# Publish 0.1.0 to PyPI
# Also add a tag for this release
$ git tag release-0.1.0 -a
$ git push origin release-0.1.0
```

For cases where the `release` branch already exists (for example when releasing 0.1.1 and `branch-0.1` already exists), just reuse that `release` branch and merge required commits from `develop` branch to the corresponding `release` branch.
