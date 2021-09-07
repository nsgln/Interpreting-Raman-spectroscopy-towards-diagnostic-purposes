# FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

# Declare some ARGuments
ARG PYTHON_VERSION=3.6
ARG CONDA_VERSION=3

# Installation of some libraries
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip python3-dev wget \
    bzip2 libopenblas-dev pbzip2 libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# INSTALLATION OF CONDA
ENV PATH /opt/conda/bin:$PATH
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda$CONDA_VERSION-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    /bin/bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm -rf /tmp/* && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
    
ADD environment.yml /tmp/environment.yml

RUN apt-get update

RUN conda update -n base -c defaults conda

RUN conda env create -f /tmp/environment.yml # Pull the environment name out of the environment.yml

RUN echo "hey"
RUN ln -sf /bin/bash /bin/sh #make sure to use bash instead of zsh

RUN apt-get update && \
	apt-get install -y apt-utils && \
    apt-get -y install sudo
     
# RUN groupadd -g $GID newuser && useradd -m -u $UID -g newuser newuser && echo "newuser:newuser" | chpasswd && adduser newuser sudo

RUN groupadd -g 5000 newuser && useradd -m -u 6520 -g newuser newuser && echo "newuser:newuser" | chpasswd && adduser newuser sudo

USER newuser

WORKDIR /home/newuser

# RUN apt install  openssh-server sudo -y

# RUN service ssh start

RUN conda init && \
    source /home/newuser/.bashrc && \
    source activate machine_learning_gpu && \
    echo "source activate $(head -1 /tmp/environment.yml | cut -d' ' -f2)" > /home/newuser/.bashrc

ENV PATH /opt/conda/envs/$(head -1 /tmp/environment.yml | cut -d' ' -f2)/bin:$PATH

EXPOSE 22

CMD /bin/bash