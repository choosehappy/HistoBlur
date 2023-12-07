# Use NVIDIA's CUDA base image
FROM nvidia/cuda:12.0.0-runtime-ubuntu22.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# System update and install basic tools
RUN apt update && \
    apt upgrade -y && \
    apt install -y software-properties-common wget bzip2 git ninja-build \
    vim nano libjpeg-dev libcairo2-dev libgdk-pixbuf2.0-dev libglib2.0-dev \
    libxml2-dev sqlite3 libopenjp2-7-dev libtiff-dev libsqlite3-dev libhdf5-dev libgl1-mesa-glx \
    build-essential && \ 
    apt clean

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH

# Install Python 3.10 using Conda
RUN conda install -c anaconda python=3.9

# Install conda packages
RUN conda install -c anaconda hdf5

# Install Meson
RUN conda install -c conda-forge meson
# Clone, compile, and install openslide
RUN git clone https://github.com/openslide/openslide.git /os-dicom
WORKDIR /os-dicom
RUN meson setup builddir && \
    meson compile -C builddir && \
    meson install -C builddir

# Update the LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH /usr/local/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
RUN ln -s /os-dicom/builddir/src/libopenslide.so.1 /usr/local/lib/x86_64-linux-gnu/libopenslide.so.0

# Install Python requirements
WORKDIR /
COPY ./ /HistoBlur/
WORKDIR /HistoBlur
RUN pip install .
WORKDIR /app
