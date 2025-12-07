#!/bin/bash

# Log everything to /var/log/user-data.log
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting user_data script..."

# 1. Mount Persistent Volume
# Wait for volume to attach
while [ ! -e /dev/nvme1n1 ]; do echo "Waiting for EBS volume..."; sleep 5; done

# Check if volume has a file system
if ! blkid /dev/nvme1n1; then
    echo "Formatting new volume..."
    mkfs -t ext4 /dev/nvme1n1
fi

mkdir -p /home/ubuntu/project_data
mount /dev/nvme1n1 /home/ubuntu/project_data
chown ubuntu:ubuntu /home/ubuntu/project_data
echo "/dev/nvme1n1 /home/ubuntu/project_data ext4 defaults,nofail 0 2" >> /etc/fstab

# 2. Setup Environment
cd /home/ubuntu/project_data

# Clone repo if not exists
if [ ! -d "neural-robot-dynamics" ]; then
    echo "Cloning repository..."
    git clone https://github.com/dhruv0000/neural-robot-dynamics.git
    chown -R ubuntu:ubuntu neural-robot-dynamics
fi

cd neural-robot-dynamics

# 3. Start Jupyter
# We use the DLAMI's pre-installed Jupyter or install if needed.
# For simplicity, we'll install jupyterlab in the base env or a venv and run it.
# The DLAMI usually has conda.

sudo -u ubuntu bash -c '
source /opt/conda/etc/profile.d/conda.sh
conda activate pytorch

pip install jupyterlab

# Generate config
jupyter lab --generate-config

# Allow remote access
echo "c.ServerApp.ip = \"0.0.0.0\"" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.allow_remote_access = True" >> ~/.jupyter/jupyter_lab_config.py
# No password for simplicity (WARNING: INSECURE)
echo "c.ServerApp.token = \"\"" >> ~/.jupyter/jupyter_lab_config.py
echo "c.ServerApp.password = \"\"" >> ~/.jupyter/jupyter_lab_config.py

# Start Jupyter in background
nohup jupyter lab --port=8888 --notebook-dir=/home/ubuntu/project_data/neural-robot-dynamics > jupyter.log 2>&1 &
'

echo "User data script completed."
