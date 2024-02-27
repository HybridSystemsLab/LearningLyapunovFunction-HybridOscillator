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
# RUN sudo apt-get install gcc python3-dev
# RUN conda install -c conda-forge gcc=12.1.0

# Install JAX
RUN conda install -c conda-forge jaxlib
RUN conda install -c conda-forge jax

# Install LaTex
# RUN apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super

# Copy the rest of your application code into the container
COPY . .

# Set the command to run your application
# CMD ["python", "your_script.py"]
RUN jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
