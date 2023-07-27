FROM nvcr.io/nvidia/tensorrt:22.08-py3

RUN apt-get update && apt-get install wget -yq
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y

WORKDIR /home

RUN git clone --recursive https://github.com/SandeepMenonKodiak/Lidar_AI_Solution.git
WORKDIR Lidar_AI_Solution/CUDA-BEVFusion

RUN apt install libprotobuf-dev -y

COPY example-data/ /home/Lidar_AI_Solution/CUDA-BEVFusion/example-data/
COPY model/ /home/Lidar_AI_Solution/CUDA-BEVFusion/model/
RUN pip install --upgrade pip
RUN pip install -r tools/requirements.txt
