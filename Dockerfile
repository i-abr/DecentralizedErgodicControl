FROM tiryoh/ros-desktop-vnc:melodic

# 安装 Miniconda
ARG CONDA_PATH=/home/miniconda
ARG CONDA_EXE=$CONDA_PATH/bin/conda
ARG PIP_EXE=$CONDA_PATH/bin/pip
ENV PATH=$CONDA_PATH/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p $CONDA_PATH && \
    rm miniconda.sh
RUN $CONDA_EXE config --set show_channel_urls yes && \
    $CONDA_EXE install -y pip && \
    $CONDA_EXE clean -ya
#    $CONDA_EXE init bash # ros依赖于系统自带的python，所以不执行conda init

RUN mkdir -p /home/catkin_ws/src && \
    cd /home/catkin_ws/src && \
    git clone https://github.com/hccz95/DecentralizedErgodicControl.git
