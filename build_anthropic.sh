#!/usr/bin/env bash
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

set -e

echo "=========================================================="
echo "    Gemini CLI - Standalone Build Script for Anthropic    "
echo "=========================================================="
echo "This script will install dependencies and create a bundled execution "
echo "environment for the Gemini CLI, including Anthropic Vertex AI support."
echo ""
echo "Building..."

# Ensure we are in the root directory
cd "$(dirname "$0")"

# Clean any previous artifacts
npm run clean

# Install dependencies including our newly added @anthropic-ai packages
npm install

# Build the bundled script
npm run bundle

echo ""
echo "=========================================================="
echo "Build complete! You can now run the CLI using the bundle:"
echo "  ./bundle/gemini.js --model claude-opus-4-6"
echo ""
echo "Make sure to set these environment variables:"
echo "  export GOOGLE_GENAI_USE_VERTEXAI=true"
echo "  export GOOGLE_CLOUD_PROJECT=<your-project-id>"
echo "  export GOOGLE_CLOUD_LOCATION=<region, e.g. us-east5>"
echo "=========================================================="
