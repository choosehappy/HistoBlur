FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04


LABEL software="HistoBlur"

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    apt-get install -y libgl1 && \
    apt-get install -y build-essential && \
    apt-get install libxrender1 && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-11.8.0/lib64:/usr/local/cuda-11.8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH


RUN conda install -c anaconda hdf5
RUN conda install -c conda-forge openslide


RUN git clone https://github.com/choosehappy/HistoBlur.git
WORKDIR /HistoBlur
RUN pip install .
WORKDIR /app

