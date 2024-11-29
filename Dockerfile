FROM python:3.12-slim

LABEL Description="FL"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y git wget unzip cmake curl git build-essential automake libtool libtool-bin

RUN mkdir /tools
WORKDIR /tools

RUN wget -O flower-main.zip https://github.com/marielonfils/flower/archive/refs/heads/main.zip
RUN unzip flower-main.zip
WORKDIR ./flower-main
RUN pip install .
WORKDIR /tools


RUN wget -O TenSEAL-main.zip https://github.com/marielonfils/TenSEAL/archive/refs/heads/main.zip
RUN unzip TenSEAL-main.zip
WORKDIR ./TenSEAL-main
RUN sed -i "s|v2.6.2|v2.11.1|g" ./cmake/pybind11.cmake
RUN pip install .
WORKDIR /

RUN pip install dill==0.3.8 GraKeL==0.1.10 matplotlib==3.8.1 matplotlib-inline==0.1.7 pandas==2.1.2 poetry==1.7.1 poetry-core==1.8.1 poetry-plugin-export==1.6.0 progressbar==2.5 protobuf==3.20.3 PyQt5==5.15.10 PyQt5-Qt5==5.15.2 PyQt5-sip==12.13.0 PyQt6==6.6.1 PyQt6-Qt6==6.6.2 PyQt6-sip==13.6.0 seaborn==0.13.0 rsa docker opencv-python-headless tornado==6.4.1
RUN pip install pyg_lib torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu


RUN pip cache purge
RUN mkdir /results
RUN mkdir /databases
RUN mkdir /ccaflr
COPY ./src /ccaflr
WORKDIR /ccaflr
RUN chmod +x FL/*.sh
RUN chmod +x FL/certificates/*.sh
RUN FL/certificates/generate_ca.sh

WORKDIR /ccaflr
