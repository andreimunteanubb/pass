#!/bin/bash

echo "Creating conda environment pass_web..."
conda env create -f environment_pass_web.yml -q

echo "Creating conda environment pass_research..."
conda env create -f environment_pass_research.yml -q

echo "Installing shfmt..."
curl -sS https://webi.sh/shfmt | sh
source ~/.config/envman/PATH.env

echo "Testing X11 forwarding..."
xeyes

echo "Container is now ready for use! Enjoy!"
