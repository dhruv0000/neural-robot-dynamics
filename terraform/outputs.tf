output "public_ip" {
  description = "Public IP of the instance"
  value       = aws_instance.nerd_server.public_ip
}

output "jupyter_url" {
  description = "URL to access Jupyter Notebook"
  value       = "http://${aws_instance.nerd_server.public_ip}:8888"
}

output "ssh_command" {
  description = "Command to SSH into the instance"
  value       = "ssh -i /path/to/${var.key_name}.pem ubuntu@${aws_instance.nerd_server.public_ip}"
}
