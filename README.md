# FCAT

[![Build Status](https://travis-ci.org/rukleiven/fcat.svg?branch=main)](https://travis-ci.org/rukleiven/fcat)

## Testing FCAT in a Container

If you want to run the CI tests in a container, FCAT is shipped with a Dockerfile for doing this.
In order to run the test using [podman](https://podman.io/)

```bash
bash podman_ci_test.sh
```

and using [docker](https://www.docker.com/)

```bash
bash docker_ci_test.sh
```