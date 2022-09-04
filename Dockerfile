FROM ubuntu:18.04

EXPOSE 9000
WORKDIR /project/

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

ENV TZ=System \ 
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y \
    build-essential \ 
    cmake \ 
    curl \ 
    gcc \ 
    git \ 
    nano \ 
    python3-dev \ 
    wget \
    zsh \ 
    && rm -rf /var/lib/apt/lists/*

ENV ZSH_THEME mortalscumbag \
    POETRY_VERSION=1.0.0

RUN /bin/sh -c chsh /bin/zsh
ENV SHELL /bin/zsh

RUN wget \ 
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh &&\
    mkdir /root/.conda &&\
    sh Miniconda3-latest-Linux-x86_64.sh -b &&\
    rm -f Miniconda3-latest-Linux-x86_64.sh 

RUN conda create --name ta22 python=3.9 -y 

COPY poetry.lock .
COPY pyproject.toml . 

SHELL ["conda", "run", "-n", "ta22", "/bin/zsh", "-c"] 
RUN conda install poetry -y && poetry install --no-ansi --no-interaction 



