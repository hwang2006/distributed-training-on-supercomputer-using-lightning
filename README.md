# Running-pytorch-lightning-on-GPU-cluster

This repository is intended to guide users to run their pytorch lightning codes on Neuron. [Pytorch Lightning](https://www.pytorchlightning.ai/index.html) is a lightweight wrapper or interface on top of PyTorch, which simplifies the implementation of complex deep learning models. It is a PyTorch extension that enables researchers and practitioners to focus more on their research and less on the engineering aspects of deep learning. PyTorch Lightning automates many of the routine tasks, such as distributing training across multiple GPUs, logging, and checkpointing, so that the users can focus on writing high-level code for their models.

**Contents**
* [KISTI Neuron GPU Cluster](#kisti-neuron-gpu-cluster)


## KISTI Neuron GPU Cluster
Neuron is a KISTI GPU cluster system consisting of 65 nodes with 260 GPUs (120 of NVIDIA A100 GPUs and 140 of NVIDIA V100 GPUs). [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling.

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205237254-b916eccc-e4b7-46a8-b7ba-c156e7609314.png"/></p>
