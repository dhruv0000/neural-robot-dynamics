variable "region" {
  description = "AWS Region"
  default     = "us-east-1"
}

variable "instance_type" {
  description = "EC2 Instance Type"
  default     = "g5.xlarge" # Fast GPU instance as requested
}

variable "key_name" {
  description = "Name of the SSH key pair to use"
  type        = string
}

variable "volume_size" {
  description = "Size of the persistent EBS volume in GB"
  default     = 100
}

variable "ami_id" {
  description = "AMI ID for Deep Learning AMI (Ubuntu 22.04). Leave empty to auto-search."
  default     = ""
}
