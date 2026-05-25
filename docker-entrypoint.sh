#!/bin/sh
set -e

# Ensure the model cache volume is owned by appuser (uid 1000).
# Docker named volumes can be initialised with root ownership, causing
# HuggingFace's chmod() calls on download temp files to fail with EPERM.
chown -R appuser:appuser /app/model_cache 2>/dev/null || true

exec runuser -u appuser -- "$@"
