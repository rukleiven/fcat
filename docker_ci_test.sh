# Bash script for building and running CI tests in podman
# It assumes that podman is available on the system

docker build -t fcat_ci_tests:latest ./
docker run -i localhost/fcat_ci_tests

# Delete the image that was created
docker rmi localhost/fcat_ci_tests:latest
