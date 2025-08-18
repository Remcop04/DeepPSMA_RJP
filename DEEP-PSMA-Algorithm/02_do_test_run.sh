#!/usr/bin/env bash

# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="example-algorithm-final"
echo "SCRIPT_DIR is: $SCRIPT_DIR"
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

echo "=+= (Re)build the container"
#source "${SCRIPT_DIR}/01_do_build.sh"

cleanup() {
    echo "=+= Cleaning permissions ..."
    # Ensure permissions are set correctly on the output
    # This allows the host user (e.g. you) to access and handle these files
    docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "$OUTPUT_DIR":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "chmod -R -f o+rwX /output/* || true"

    # Ensure volume is removed
    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null 2>&1 || true
}

# This allows for the Docker user to read
chmod -R -f o+rX "$INPUT_DIR" || true
if [ -d "${SCRIPT_DIR}/model" ]; then
    chmod -R -f o+rX "${SCRIPT_DIR}/model" || true
fi

if [ -d "${OUTPUT_DIR}/interf0" ]; then
  # This allows for the Docker user to write
  chmod -f o+rwX "${OUTPUT_DIR}/interf0" || true

  echo "=+= Cleaning up any earlier output"
  # Use the container itself to circumvent ownership problems
  docker run --rm \
      --platform=linux/amd64 \
      --quiet \
      --volume "${OUTPUT_DIR}/interf0":/output \
      --entrypoint /bin/sh \
      $DOCKER_IMAGE_TAG \
      -c "rm -rf /output/* || true"
else
  mkdir -p "${OUTPUT_DIR}/interf0"
  chmod o+rwX "${OUTPUT_DIR}/interf0" || true
fi

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null 2>&1 || true

trap cleanup EXIT

run_docker_forward_pass() {
    local interface_dir="$1"

    echo "=+= Doing a forward pass on ${interface_dir} (CPU-only mode)"

    # REMOVED: --gpus all (this was causing the error)
    # ADDED: CPU device specification in environment variable
    docker run --rm \
        --platform=linux/amd64 \
        --network none \
        --shm-size 2g \
        --env CUDA_VISIBLE_DEVICES="" \
        --volume "${INPUT_DIR}/${interface_dir}":/input:ro \
        --volume "${OUTPUT_DIR}/${interface_dir}":/output \
        --volume "$DOCKER_NOOP_VOLUME":/tmp \
        --volume "${SCRIPT_DIR}/model":/opt/ml/model:ro \
        "$DOCKER_IMAGE_TAG"

  echo "=+= Wrote results to ${OUTPUT_DIR}/${interface_dir}"
}

run_docker_forward_pass "interf0"

echo "=+= Save this image for uploading via ./03_do_save.sh"