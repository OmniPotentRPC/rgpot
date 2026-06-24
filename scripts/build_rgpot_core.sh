#!/usr/bin/env bash
set -euo pipefail

manifest_path=$1
target_dir=$2
profile_dir=$3
output_path=$4
feature_list=${5:-}

cargo_args=(build --manifest-path "$manifest_path" --target-dir "$target_dir")
if [[ "$profile_dir" == "release" ]]; then
  cargo_args+=(--release)
fi
if [[ -n "$feature_list" ]]; then
  cargo_args+=(--features "$feature_list")
fi

cargo "${cargo_args[@]}"
cp "$target_dir/$profile_dir/librgpot_core.a" "$output_path"
