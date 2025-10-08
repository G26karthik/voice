#!/bin/bash

# Install spaCy English model
python -m spacy download en_core_web_sm

# Create necessary directories
mkdir -p models
mkdir -p .cache

echo "✅ Setup complete!"
