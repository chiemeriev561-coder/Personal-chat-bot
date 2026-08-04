#!/usr/bin/env bash
set -euo pipefail

# Never store API keys in the project. Run this file with `source setenv.sh`
# after exporting GEMINI_API_KEY (or GOOGLE_API_KEY) in your shell.
if [[ -z "${GEMINI_API_KEY:-}" && -z "${GOOGLE_API_KEY:-}" ]]; then
	printf 'Set GEMINI_API_KEY or GOOGLE_API_KEY before sourcing this file.\n' >&2
	if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
		return 1
	fi
	exit 1
fi

printf 'API key found in the environment.\n'
