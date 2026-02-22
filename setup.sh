#!/bin/bash
# filepath: setup.sh

echo "=== Install system dependencies ==="
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1 libsndfile1-dev libavcodec-extra

echo "=== Install python dependencies ==="
pip install --upgrade pip
pip install -r requirements.txt
