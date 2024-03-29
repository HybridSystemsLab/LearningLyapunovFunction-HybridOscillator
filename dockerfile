# Use the official Anaconda base image
FROM continuumio/anaconda3:latest

# Set the working directory inside the container
WORKDIR /app

# Copy the environment file into the container
COPY environment.yml .

# Create the Conda environment
RUN conda env create -f environment.yml

# Activate the Conda environment
RUN echo "source activate $(head -1 environment.yml | cut -d' ' -f2)" > ~/.bashrc
ENV PATH /opt/conda/envs/$(head -1 environment.yml | cut -d' ' -f2)/bin:$PATH

# Install additional packages
RUN conda install -c conda-forge jupyter
# RUN conda install -c conda-forge gcc=12.1.0

# Install JAX
RUN conda install -c conda-forge jaxlib=0.3.25
RUN conda install -c conda-forge jax=0.3.25

# Install LaTex
# RUN apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super
RUN apt-get update && apt-get install -y \
    # texlive \
    texlive-full \
    # texlive-latex-extra \
    # texlive-fonts-recommended \
    # cm-super \
    dvipng

# Attempt to install texlive-latex-extra, but ignore errors if it fails
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     && apt-cache show texlive-latex-extra > /dev/null \
#     && apt-get install -y texlive-latex-extra \
#     || echo "Skipping texlive-latex-extra installation due to failure."

# Copy the rest of your application code into the container
COPY . .