#!/usr/bin/env bash
# Bump the version, commit, tag vX.Y.Z and push. The tag is the release: gnome-pipelines pins
# gnomepy-research by it (gnomepy_research/models/ is what it imports).
set -euo pipefail

bump="${1:-}"
case "$bump" in
  patch|minor|major) ;;
  *) echo "usage: scripts/release.sh patch|minor|major" >&2; exit 2 ;;
esac

cd "$(git rev-parse --show-toplevel)"

branch=$(git rev-parse --abbrev-ref HEAD)
if [ "$branch" != "main" ]; then
  echo "release from main (currently on $branch)" >&2
  exit 1
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "working tree has uncommitted changes" >&2
  exit 1
fi
git fetch --quiet origin main
if [ "$(git rev-parse HEAD)" != "$(git rev-parse origin/main)" ]; then
  echo "main is not in sync with origin/main; pull or push first" >&2
  exit 1
fi

poetry version "$bump"
version=$(poetry version -s)
tag="v$version"
if git rev-parse --quiet --verify "refs/tags/$tag" >/dev/null; then
  git checkout --quiet -- pyproject.toml
  echo "tag $tag already exists" >&2
  exit 1
fi

git commit --quiet -m "release: $tag" -- pyproject.toml
git tag -a "$tag" -m "$tag"

read -r -p "push $tag to origin? [y/N] " answer
if [ "$answer" != "y" ]; then
  echo "not pushed. undo with: git tag -d $tag && git reset --hard HEAD~1"
  exit 1
fi
# --atomic: the commit and the tag land together or not at all.
git push --atomic origin main "$tag"
echo "pushed $tag; point gnome-pipelines at it: gnomepy-research = {git = \"https://github.com/gnome-trading-group/gnomepy-research.git\", tag = \"$tag\"}"
