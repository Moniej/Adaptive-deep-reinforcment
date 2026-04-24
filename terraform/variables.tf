variable "aws_region" {
  description = "AWS region for deployment."
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name used in resource names and tags."
  type        = string
  default     = "adaptive-drl-forex"
}

variable "environment" {
  description = "Deployment environment label."
  type        = string
  default     = "dev"
}

variable "instance_type" {
  description = "EC2 instance type for training/evaluation."
  type        = string
  default     = "t3.small"
}

variable "root_volume_size" {
  description = "Root EBS volume size in GB."
  type        = number
  default     = 30
}

variable "project_bucket_name" {
  description = "Globally unique S3 bucket name for code artifacts and results."
  type        = string
}

variable "key_pair_name" {
  description = "Existing AWS EC2 key pair name for SSH access."
  type        = string
}

variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed to SSH into the EC2 instance."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
