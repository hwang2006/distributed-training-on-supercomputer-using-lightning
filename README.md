# Distributed-training-with-pytorch-lightning

This repository is intended to guide users to run their pytorch lightning codes on Neuron. [Pytorch Lightning](https://www.pytorchlightning.ai/index.html) is a lightweight wrapper or interface on top of PyTorch, which simplifies the implementation of complex deep learning models. It is a PyTorch extension that enables researchers and practitioners to focus more on their research and less on the engineering aspects of deep learning. PyTorch Lightning automates many of the routine tasks, such as distributing training across multiple GPUs, logging, and checkpointing, so that the users can focus on writing high-level code for their models.

**Contents**
* [KISTI Neuron GPU Cluster](#kisti-neuron-gpu-cluster)
* [Installing Conda](#installing-conda)
* [Installing Pytorch Lightning](#installing-pytorch-lightning)
* [Running Jupyter](#running-jupyter)
* [Pytorch Lightning Simple Examples on Jupyter](#pytorch-lightning-simple-examples-on-jupyter) 


## KISTI Neuron GPU Cluster
Neuron is a KISTI GPU cluster system consisting of 65 nodes with 260 GPUs (120 of NVIDIA A100 GPUs and 140 of NVIDIA V100 GPUs). [Slurm](https://slurm.schedmd.com/) is adopted for cluster/resource management and job scheduling.

<p align="center"><img src="https://user-images.githubusercontent.com/84169368/205237254-b916eccc-e4b7-46a8-b7ba-c156e7609314.png"/></p>

## Installing Conda
Once logging in to Neuron, you will need to have either [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your scratch directory. Anaconda is distribution of the Python and R programming languages for scientific computing, aiming to simplify package management and deployment. Anaconda comes with +150 data science packages, whereas Miniconda, a small bootstrap version of Anaconda, comes with a handful of what's needed.

1. Check the Neuron system specification
```
[glogin01]$ cat /etc/*release*
CentOS Linux release 7.9.2009 (Core)
Derived from Red Hat Enterprise Linux 7.8 (Source)
NAME="CentOS Linux"
VERSION="7 (Core)"
ID="centos"
ID_LIKE="rhel fedora"
VERSION_ID="7"
PRETTY_NAME="CentOS Linux 7 (Core)"
ANSI_COLOR="0;31"
CPE_NAME="cpe:/o:centos:centos:7"
HOME_URL="https://www.centos.org/"
BUG_REPORT_URL="https://bugs.centos.org/"

CENTOS_MANTISBT_PROJECT="CentOS-7"
CENTOS_MANTISBT_PROJECT_VERSION="7"
REDHAT_SUPPORT_PRODUCT="centos"
REDHAT_SUPPORT_PRODUCT_VERSION="7"

CentOS Linux release 7.9.2009 (Core)
CentOS Linux release 7.9.2009 (Core)
cpe:/o:centos:centos:7
```

2. Download Anaconda or Miniconda. Miniconda comes with python, conda (package & environment manager), and some basic packages. Miniconda is fast to install and could be sufficient for distributed deep learning training practices. 
```
# (option 1) Anaconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
```
```
# (option 2) Miniconda 
[glogin01]$ cd /scratch/$USER  ## Note that $USER means your user account name on Neuron
[glogin01]$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

3. Install Miniconda. By default conda will be installed in your home directory, which has a limited disk space. You will install and create subsequent conda environments on your scratch directory. 
```
[glogin01]$ chmod 755 Miniconda3-latest-Linux-x86_64.sh
[glogin01]$ ./Miniconda3-latest-Linux-x86_64.sh

Welcome to Miniconda3 py39_4.12.0

In order to continue the installation process, please review the license
agreement.
Please, press ENTER to continue
>>>                               <======== press ENTER here
.
.
.
Do you accept the license terms? [yes|no]
[no] >>> yes                      <========= type yes here 

Miniconda3 will now be installed into this location:
/home01/qualis/miniconda3        

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/home01/qualis/miniconda3] >>> /scratch/$USER/miniconda3  <======== type /scratch/$USER/miniconda3 here
PREFIX=/scratch/qualis/miniconda3
Unpacking payload ...
Collecting package metadata (current_repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /scratch/qualis/miniconda3
.
.
.
Preparing transaction: done
Executing transaction: done
installation finished.
Do you wish the installer to initialize Miniconda3
by running conda init? [yes|no]
[no] >>> yes         <========== type yes here
.
.
.
If you'd prefer that conda's base environment not be activated on startup,
   set the auto_activate_base parameter to false:

conda config --set auto_activate_base false

Thank you for installing Miniconda3!
```

4. finalize installing Miniconda with environment variables set including conda path

```
[glogin01]$ source ~/.bashrc    # set conda path and environment variables 
[glogin01]$ conda config --set auto_activate_base false
[glogin01]$ which conda
/scratch/$USER/miniconda3/condabin/conda
[glogin01]$ conda --version
conda 4.12.0
```

## Installing Pytorch Lightning
Now you are ready to build Horovod as a conda virtual environment: 
1. load modules: 
```
module load gcc/10.2.0 cuda/11.7
```
2. create a new conda virtual environment and activate the environment:
```
[glogin01]$ conda create -n lightning
[glogin01]$ conda activate lightning
```
3. install the pytorch and lightning package:
```
(lightning) [glogin01]$ conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
(lightning) [glogin01]$ pip install lightning
```
4. check if the pytorch lightning packages were installed:
```
(lightning) [glogin01]$ conda list | grep lightning
# packages in environment at /scratch/$USER/miniconda3/envs/lightning:
lightning                 2.0.2                    pypi_0    pypi
lightning-cloud           0.5.36                   pypi_0    pypi
lightning-utilities       0.8.0                    pypi_0    pypi
pytorch-lightning         2.0.2                    pypi_0    pypi
```

## Running Jupyter
[Jupyter](https://jupyter.org/) is free software, open standards, and web services for interactive computing across all programming languages. Jupyterlab is the latest web-based interactive development environment for notebooks, code, and data. The Jupyter Notebook is the original web application for creating and sharing computational documents. You will run a notebook server on a worker node (*not* on a login node), which will be accessed from the browser on your PC or labtop through SSH tunneling. 
<p align="center"><img src="https://github.com/hwang2006/KISTI-DL-tutorial-using-horovod/assets/84169368/34a753fc-ccb7-423e-b0f3-f973b8cd7122"/>
</p>

In order to do so, you need to add the horovod-enabled virtual envrionment that you have created as a python kernel.
1. activate the horovod-enabled virtual environment:
```
[glogin01]$ conda activate lightning
```
2. install Jupyter on the virtual environment:
```
(lightning) [glogin01]$ conda install jupyter
(lightning) [glogin01]$ pip install jupyter-tensorboard
```
3. add the virtual environment as a jupyter kernel:
```
(lightning) [glogin01]$ pip install ipykernel 
(lightning) [glogin01]$ python -m ipykernel install --user --name lightning
```
4. check the list of kernels currently installed:
```
(lightning) [glogin01]$ jupyter kernelspec list
Available kernels:
python3       /home01/$USER/.local/share/jupyter/kernels/python3
horovod       /home01/$USER/.local/share/jupyter/kernels/lightning
```
5. launch a jupyter notebook server on a worker node 
- to deactivate the virtual environment
```
(horovod) [glogin01]$ conda deactivate
```
- to create a batch script for launching a jupyter notebook server: 
```
[glogin01]$ cat jupyter_run.sh
#!/bin/bash
#SBATCH --comment=tensorflow
##SBATCH --partition=mig_amd_a100_4
#SBATCH --partition=amd_a100nv_8
#SBATCH --time=8:00:00        # walltime
#SBATCH --nodes=1             # the number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --gres=gpu:1          # number of gpus per node
#SBATCH --cpus-per-task=4     # number of cpus per task

#removing the old port forwading
if [ -e port_forwarding_command ]
then
  rm port_forwarding_command
fi

#getting the node name and port
SERVER="`hostname`"
PORT_JU=$(($RANDOM + 10000 )) # some random number greaten than 10000

echo $SERVER
echo $PORT_JU

echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
echo "ssh -L localhost:8888:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr" > port_forwarding_command
#echo "ssh -L localhost:${PORT_JU}:${SERVER}:${PORT_JU} ${USER}@neuron.ksc.re.kr"

echo "load module-environment"
module load gcc/10.2.0 cuda/11.4 cudampi/openmpi-4.1.1

echo "execute jupyter"
source ~/.bashrc
conda activate lightning
cd /scratch/$USER     # the root/work directory of the jupyter lab/notebook to be launched
jupyter lab --ip=0.0.0.0 --port=${PORT_JU} --NotebookApp.token=${USER} #jupyter token: your user name
echo "end of the job"
```
- to launch a jupyter notebook server 
```
[glogin01]$ sbatch jupyter_run.sh
Submitted batch job XXXXXX
```
- to check if a jupyter notebook server is running
```
[glogin01]$ squeue -u $USER
             JOBID       PARTITION     NAME     USER    STATE       TIME TIME_LIMI  NODES NODELIST(REASON)
            XXXXXX    amd_a100nv_8 jupyter_    $USER  RUNNING       0:02   8:00:00      1 gpu30
[glogin01]$ cat slurm-XXXXXX.out
.
.
[I 2023-02-14 08:30:04.790 ServerApp] Jupyter Server 1.23.4 is running at:
[I 2023-02-14 08:30:04.790 ServerApp] http://gpu##:#####/lab?token=...
.
.
```
- to check the SSH tunneling information generated by the jupyter_run.sh script 
```
[glogin01]$ cat port_forwarding_command
ssh -L localhost:8888:gpu##:##### $USER@neuron.ksc.re.kr
```
6. open a SSH client (e.g., Putty, PowerShell, Command Prompt, etc) on your PC or laptop and log in to Neuron just by copying and pasting the port_forwarding_command:
```
C:\Users\hwang>ssh -L localhost:8888:gpu##:##### $USER@neuron.ksc.re.kr
Password(OTP):
Password:
```
7. open a web browser on your PC or laptop to access the jupyter server
```
URL Address: localhost:8888e
Password or token: $USER    # your account ID on Neuron
```
<p align="center"><img src="https://user-images.githubusercontent.com/84169368/218938419-f38c356b-e682-4b1c-9add-6cfc29d53425.png"/></p> 

## Pytorch Lightning Simple Examples on Jupyter
Now, you are ready to run a pytorch lightning code on a jupyter notebook. Hopefully, the jupyter notebook examples would lead to getting familiarized yourself to the basics of pytorch lightning coding practices step-by-step. Refer to the [notebooks]((https:https://github.com/hwang2006/distributed-training-with-pytorch-lightning/tree/main/notebooks) directory for example codes.
* [Link to Jupyter Notebook Example](https://nbviewer.org/github/hwang2006/distributed-training-with-pytorch-lightning/blob/main/notebooks/pytorch_lightning_example.ipynb)




