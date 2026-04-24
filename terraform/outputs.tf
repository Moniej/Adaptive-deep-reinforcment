output "instance_public_ip" {
  description = "Public IP of the EC2 instance."
  value       = aws_instance.trading_vm.public_ip
}

output "instance_public_dns" {
  description = "Public DNS name of the EC2 instance."
  value       = aws_instance.trading_vm.public_dns
}

output "s3_bucket_name" {
  description = "S3 bucket for datasets and results."
  value       = aws_s3_bucket.project_bucket.bucket
}

output "ssh_command" {
  description = "SSH command to connect to the EC2 instance."
  value       = "ssh -i <path-to-your-private-key> ec2-user@${aws_instance.trading_vm.public_dns}"
}
