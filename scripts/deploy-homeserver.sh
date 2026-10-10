#!/bin/bash
# Deploy a specific main commit through the restricted SSH key.
set -euo pipefail

repo_dir="${1:?Missing repository path}"
commit="${SSH_ORIGINAL_COMMAND:-}"

if [[ ! "$commit" =~ ^[0-9a-f]{40}$ ]]; then
    echo "Expected a commit SHA." >&2
    exit 1
fi

cd "$repo_dir"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "Repository has local changes; deployment aborted." >&2
    exit 1
fi

git fetch origin main

if [[ "$commit" != "$(git rev-parse origin/main)" ]]; then
    echo "Commit is no longer the latest main; deployment aborted." >&2
    exit 1
fi

git checkout --no-overwrite-ignore --detach "$commit"

docker compose config --quiet
docker compose build api frontend
docker compose up -d --no-deps --force-recreate api frontend

curl --fail --silent --show-error \
    --retry 10 --retry-delay 3 --retry-all-errors \
    --max-time 10 http://127.0.0.1:8081/api/ready

curl --fail --silent --show-error \
    --retry 10 --retry-delay 3 --retry-all-errors \
    --max-time 10 http://127.0.0.1:8081/ > /dev/null
