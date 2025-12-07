# Terraform Infrastructure for Neural Robot Dynamics

This directory contains Terraform configuration to provision an AWS EC2 instance for running Jupyter Notebooks with persistent storage.

## Prerequisites

1.  **Terraform**: Install Terraform from [terraform.io](https://www.terraform.io/downloads).
2.  **AWS CLI**: Install and configure AWS CLI with your credentials (`aws configure`).
3.  **SSH Key Pair**: Ensure you have an SSH key pair in your AWS region (default `us-east-1`).

## Usage

1.  **Initialize Terraform**:
    ```bash
    cd terraform
    terraform init
    ```

2.  **Plan Infrastructure**:
    Check what resources will be created. You need to provide your key pair name.
    ```bash
    terraform plan -var="key_name=your-key-name"
    ```

3.  **Apply Infrastructure**:
    Provision the resources.
    ```bash
    terraform apply -var="key_name=your-key-name"
    ```
    Type `yes` to confirm.

4.  **Access Jupyter**:
    After a few minutes (for the instance to boot and script to run), Terraform will output the `jupyter_url`.
    Open that URL in your browser.
    *Note: The setup script configures Jupyter without a password for simplicity. Ensure your Security Group restricts access if needed.*

5.  **SSH Access**:
    Terraform also outputs the `ssh_command`.
    ```bash
    ssh -i /path/to/your-key.pem ubuntu@<public-ip>
    ```

## Managing the Instance

**To Stop the Instance (Save Money):**
You can stop the instance via the AWS Console or CLI. The EBS volume (`/home/ubuntu/project_data`) will persist.
```bash
aws ec2 stop-instances --instance-ids <instance-id>
```

**To Start the Instance:**
```bash
aws ec2 start-instances --instance-ids <instance-id>
```
*Note: The Public IP will change after a stop/start unless you use an Elastic IP.*

**To Destroy Everything (When finished with the project):**
```bash
terraform destroy -var="key_name=your-key-name"
```
*Warning: This will delete the persistent EBS volume and all data on it.*

## Configuration

You can customize variables in `terraform/variables.tf` or pass them via `-var`:
- `region`: AWS Region (default `us-east-1`).
- `instance_type`: EC2 Instance Type (default `g5.xlarge`).
- `volume_size`: Size of persistent storage (default `100` GB).

## Estimated Costs

Prices are approximate for `us-east-1` (N. Virginia) as of late 2024.

### 1. Infrastructure Idle Cost (Instance Stopped)
When the instance is stopped, you only pay for the persistent storage.
*   **EBS Storage (gp3)**: ~$0.08 per GB-month.

| Disk Size | Cost (Approx) | Notes |
| :--- | :--- | :--- |
| **50 GB** | **~$4.00 / month** | Minimum recommended for OS + Code. |
| **100 GB** | **~$8.00 / month** | **Default**. Good for 10M transitions. |
| **200 GB** | **~$16.00 / month** | If storing multiple large datasets. |
| **500 GB** | **~$40.00 / month** | Heavy data usage. |

### 2. Active Compute Cost (Instance Running)
When the instance is running, you pay for the compute time + storage.

| Instance Type | GPU | VRAM | Cost (Approx) | Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **g4dn.xlarge** | NVIDIA T4 | 16GB | **~$0.526 / hour** | Budget, same as Colab free/Pro. |
| **g5.xlarge** | NVIDIA A10G | 24GB | **~$1.006 / hour** | **Recommended**. Good speed/price balance. |
| **p3.2xlarge** | NVIDIA V100 | 16GB | **~$3.06 / hour** | Fast training, expensive. |

**Recommendation:** Stop the instance whenever you are not actively training or coding to minimize costs.
