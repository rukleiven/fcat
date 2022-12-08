# Bash script for building and running CI tests in podman
# It assumes that podman is available on the system

podman build -t fcat_ci_tests:latest ./
podman run --rm -it localhost/fcat_ci_tests

# Delete the image that was created
podman rmi localhost/fcat_ci_tests:latest
