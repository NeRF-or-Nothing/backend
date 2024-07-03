#!/bin/bash

# This script is the entry point for the nerf-worker-gaussian container.
# It installs the submodules that need nvcc to build, builds them, 
# and then runs the main.py Python script you would normally find in 
# the other service containers like sfm-worker.

echo "Starting entry_point.sh"

# Function to check if a package is installed
is_package_installed() {
    pip show $1 > /dev/null 2>&1
    return $?
}

# Build and install simple-knn if not already installed
if ! is_package_installed "simple-knn"; then
    echo "Building and installing simple-knn"
    cd submodules/simple-knn
    python setup.py bdist_wheel
    pip install dist/*
    cd ../..
    echo "Finished building and installing simple-knn"
else
    echo "simple-knn is already installed"
fi

# Build and install diff-gaussian-rasterization if not already installed
if ! is_package_installed "diff_gaussian_rasterization"; then
    echo "Building and installing diff-gaussian-rasterization"
    cd submodules/diff-gaussian-rasterization
    python setup.py bdist_wheel
    pip install dist/*
    cd ../..
    echo "Finished building and installing diff-gaussian-rasterization"
else
    echo "diff-gaussian-rasterization is already installed"
fi

# Run the main Python script
echo "Running main Python script"
python train.py -s data/sfm_data/TestUUID/ -m data/nerf_data/TestUUID