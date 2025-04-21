FROM 192.168.18.237/pvmed/nvidia/cuda:12.3.2-devel-ubuntu22.04

ENV PYTHONUNBUFFERED 1
ENV PYTHONWARNINGS="ignore"

RUN apt-get update 
RUN apt-get install -y software-properties-common
RUN apt-get update 
RUN apt-get install -y python3.10 python3.10-distutils python3-pip
RUN apt-get -f install --fix-missing
RUN apt-get install -y libgl1-mesa-glx

COPY ./ ./

RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /mednext
RUN pip3 install -e .
WORKDIR /


CMD ["python3", "inference.py", "config_files/config_infer.yaml"]