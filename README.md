# Mjölnir User Documentation

Welcome to the Ruqola project NTU research server (aka Mjölnir)! This documentation provides comprehensive guidance for using our shared GPU computing resources effectively.

You can access a Jeckyll version of this documentation [here](https://ighina.github.io/ruqola-server-deploy/).

## 🖥️ Server Specifications

- **GPUs**: 3x NVIDIA H200 (80GB HBM3e each)
- **Total GPU Memory**: 240GB
- **Custom Queue Management**: Fair resource allocation system

## 📬 Notifications

The server is set up to send notifications to individual users and/or to administrator and general channels. If you receive such a notification or you want to familiarize yourself with the type of notifications the server sends, consult the following [**notification FAQ**](docs/notifications-faq.md).

## 📚 Documentation Structure

### For New Users
- [**Bash Basics**](docs/bash-basics.md) - Essential command line skills for server usage
- [**Server Best Practices**](docs/best-practices.md) - Guidelines for respectful resource sharing

### Server Users Creation and Deletion
- [**Create/Delete Users**](docs/users-creation.md) - Explain the two minimal scripts used to create and delete users either individually or from a csv file containing multiple users.

### File System and Folders Structure
- [**Users Quota**](docs/users-quota.md) - Explain the quotas system used to limit the disk space used by each user and some essential bash commands to check and manage quotas.  
- [**Scratch Folder**](docs/scratch-folder.md) - Guidelines and information for use the scratch folder to store large datasets and programmes artifacts.

### GPU Queue System
- [**GPU Queue User Guide**](docs/gpu-queue-guide.md) - Comprehensive guide to job submission and monitoring
- [**H200 GPU Specifications**](docs/h200-specs.md) - Technical details and capabilities
- [**Custom Queue Setup**](gpuq/README.md) - Technical setup and administration (existing)

### Deep Learning Frameworks
- [**PyTorch with H200**](docs/pytorch-guide.md) - Optimized PyTorch usage and examples
- [**TensorFlow/Keras with H200**](docs/tensorflow-guide.md) - TensorFlow setup and best practices
- [**JAX/Flax with H200**](docs/jax-guide.md) - JAX configuration and usage patterns
- [**Transformers with H200**](docs/transformers-guide.md) - Hugging Face Transformers for LLMs and fine-tuning

### Examples and Scripts
- [**Example Scripts**](examples/) - Ready-to-use scripts for common workflows
- [**Troubleshooting**](docs/troubleshooting.md) - Common issues and solutions

## 🚀 Quick Start

1. **First Time Setup**: Read [Bash Basics](docs/bash-basics.md)
2. **Familiarise yourself with file and folder structure**: Read [Users Quota](docs/users-quota.md) and [Scratch Folder](scratch-folder.md)
3. **Submit Your First Job**: Check [GPU Queue Guide](docs/gpu-queue-guide.md)
4. **Choose Your Framework**: Select from PyTorch, TensorFlow, or JAX guides
5. **Optimize Your Code**: Review [Best Practices](docs/best-practices.md)

## ⚡ Quick Commands

```bash
# Check GPU availability
gpuq status

# Submit a training job
conda activate $your_environment
gpuq submit --command "python train.py" --gpus 1 --time 8

# Monitor GPUs in real-time
nvidia-smi -l 1

# Check your running jobs
gpuq status | grep $USER
```

## 📞 Getting Help

- **Technical Issues**: Contact your server administrator
- **Documentation Updates**: Submit suggestions or corrections
- **Queue System**: Check [gpuq/README.md](gpuq/README.md) for technical details

---

*Last updated: September 2025*