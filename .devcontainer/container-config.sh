#!/bin/bash

echo "Creating conda environment webui..."
conda env create -f webui/environment_webui.yml -q

echo "Creating conda environment research..."
conda env create -f research/environment_research.yml -q

echo "Installing shfmt..."
curl -sS https://webi.sh/shfmt | sh
source ~/.config/envman/PATH.env

echo "Testing X11 forwarding..."
xeyes

echo "Container is now ready for use! Enjoy!"
