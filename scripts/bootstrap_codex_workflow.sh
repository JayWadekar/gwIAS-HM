#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/bootstrap_codex_workflow.sh --target /path/to/other/repo [--copy-hooks] [--force]

What it copies:
  - AGENTS.md
  - .agent/spec/* (pipeline orientation/spec/todo)
  - docs/sphinx scaffold (source, script, Makefile, requirements)
  - .github/workflows/docs.yml

Options:
  --target PATH   Target git repository root (required)
  --copy-hooks    Also copy local pre-commit and pre-push hooks into target .git/hooks
  --force         Overwrite existing files in target repo
  -h, --help      Show this help
EOF
}

TARGET=""
COPY_HOOKS=0
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      TARGET="${2:-}"
      shift 2
      ;;
    --copy-hooks)
      COPY_HOOKS=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$TARGET" ]]; then
  echo "--target is required." >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="$(cd "$TARGET" && pwd)"

if [[ ! -d "$TARGET/.git" ]]; then
  echo "Target is not a git repository: $TARGET" >&2
  exit 1
fi

copy_file() {
  local src="$1"
  local dst="$2"
  local mode="${3:-644}"

  if [[ ! -f "$src" ]]; then
    echo "Missing source file: $src" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$dst")"
  if [[ -e "$dst" && "$FORCE" -ne 1 ]]; then
    echo "Skip (exists): $dst"
    return
  fi
  install -m "$mode" "$src" "$dst"
  echo "Copied: $dst"
}

copy_file "$SRC_REPO_ROOT/AGENTS.md" "$TARGET/AGENTS.md"
copy_file "$SRC_REPO_ROOT/.agent/spec/PIPELINE_SPEC.md" "$TARGET/.agent/spec/PIPELINE_SPEC.md"
copy_file "$SRC_REPO_ROOT/.agent/spec/PIPELINE_ORIENTATION.md" "$TARGET/.agent/spec/PIPELINE_ORIENTATION.md"
copy_file "$SRC_REPO_ROOT/.agent/spec/PIPELINE_TODO.md" "$TARGET/.agent/spec/PIPELINE_TODO.md"

copy_file "$SRC_REPO_ROOT/docs/sphinx/Makefile" "$TARGET/docs/sphinx/Makefile"
copy_file "$SRC_REPO_ROOT/docs/sphinx/README.md" "$TARGET/docs/sphinx/README.md"
copy_file "$SRC_REPO_ROOT/docs/sphinx/requirements.txt" "$TARGET/docs/sphinx/requirements.txt"
copy_file "$SRC_REPO_ROOT/docs/sphinx/scripts/generate_api_tree.py" "$TARGET/docs/sphinx/scripts/generate_api_tree.py" 755
copy_file "$SRC_REPO_ROOT/docs/sphinx/source/conf.py" "$TARGET/docs/sphinx/source/conf.py"
copy_file "$SRC_REPO_ROOT/docs/sphinx/source/index.rst" "$TARGET/docs/sphinx/source/index.rst"
copy_file "$SRC_REPO_ROOT/docs/sphinx/source/getting_started.rst" "$TARGET/docs/sphinx/source/getting_started.rst"
copy_file "$SRC_REPO_ROOT/docs/sphinx/source/api/index.rst" "$TARGET/docs/sphinx/source/api/index.rst"
copy_file "$SRC_REPO_ROOT/docs/sphinx/source/api/full_api.rst" "$TARGET/docs/sphinx/source/api/full_api.rst"
copy_file "$SRC_REPO_ROOT/docs/sphinx/source/_static/api.css" "$TARGET/docs/sphinx/source/_static/api.css"

copy_file "$SRC_REPO_ROOT/.github/workflows/docs.yml" "$TARGET/.github/workflows/docs.yml"

if [[ "$COPY_HOOKS" -eq 1 ]]; then
  if [[ -f "$SRC_REPO_ROOT/.git/hooks/pre-commit" ]]; then
    copy_file "$SRC_REPO_ROOT/.git/hooks/pre-commit" "$TARGET/.git/hooks/pre-commit" 755
  fi
  if [[ -f "$SRC_REPO_ROOT/.git/hooks/pre-push" ]]; then
    copy_file "$SRC_REPO_ROOT/.git/hooks/pre-push" "$TARGET/.git/hooks/pre-push" 755
  fi
fi

cat <<EOF

Bootstrap complete for: $TARGET

Next steps in target repo:
  1) Review AGENTS.md and .agent/spec/* for project-specific wording.
  2) Update docs/sphinx/source/conf.py project metadata and import paths if needed.
  3) If your source code is not in "Pipeline/", edit docs/sphinx/scripts/generate_api_tree.py and conf.py accordingly.
  4) Run:
       python3 docs/sphinx/scripts/generate_api_tree.py
       (cd docs/sphinx && make html)
EOF
