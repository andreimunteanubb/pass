#!/bin/bash

echo "Creating conda environment pass_web..."
conda env create -f environment_pass_web.yml -q

echo "Creating conda environment pass_research..."
conda env create -f environment_pass_research.yml -q

echo "Installing shfmt..."
curl -sS https://webi.sh/shfmt | sh
source ~/.config/envman/PATH.env

#read -p "Enter your GitHub username: " github_username
#github_username="andreimunteanubb"
#git config --global user.name "$github_username"

#read -p "Enter your GitHub email: " github_email
#github_email="andrei.muntean@stud.ubbcluj.ro"
#git config --global user.email "$github_email"

echo "Testing X11 forwarding..."
xeyes

echo "Container is now ready for use! Enjoy!"
