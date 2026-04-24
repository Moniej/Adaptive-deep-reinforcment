# AWS Terraform Setup

This folder provisions:

- `1` EC2 instance for training and evaluation
- `1` private S3 bucket for datasets and results
- a security group for SSH access
- an IAM role so the VM can read/write to the S3 bucket

## Before you start

1. Install Terraform.
2. Install and configure the AWS CLI.
3. Make sure you already have an EC2 key pair in your AWS account.

## Files to edit

1. Copy `terraform.tfvars.example` to `terraform.tfvars`
2. Fill in:
   - `project_bucket_name`
   - `key_pair_name`
   - optionally `allowed_ssh_cidrs`

## Commands

Run these from this `terraform` folder:

```powershell
terraform init
terraform plan
terraform apply
```

After apply, Terraform will print:

- EC2 public IP
- EC2 public DNS
- S3 bucket name
- SSH command template

## After provisioning

SSH into the instance:

```powershell
ssh -i <path-to-your-private-key> ec2-user@<instance-public-dns>
```

Then upload your project to the VM or clone it there, install Python dependencies, and run your training script.

## Cleanup

To avoid charges:

```powershell
terraform destroy
```
