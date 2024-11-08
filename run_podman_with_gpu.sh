#!/usr/bin/env bash

BASEDIR="$(dirname $(realpath ${0}))/"
podman run --rm -it --device nvidia.com/gpu=all --security-opt=label=disable \
  -v ${BASEDIR}/bucket_in_folder:/home/nibio/mutable-outside-world/bucket_in_folder:z \
  -v ${BASEDIR}/bucket_out_folder:/home/nibio/mutable-outside-world/bucket_out_folder:z \
  localhost/nibio/e2e-oracle-inst-seg:latest
