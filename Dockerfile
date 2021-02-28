ARG GPU=OFF
ARG IMAGE=ubuntu:18.04

FROM $IMAGE

WORKDIR /root

COPY . .

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Python package management and basic dependencies
RUN apt-get install -y curl  python3.6-dev python3.6-distutils
# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.6 1
# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.6
# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

RUN DEBIAN_FRONTEND=noninteractive  apt-get --yes install build-essential checkinstall
RUN DEBIAN_FRONTEND=noninteractive  apt-get --yes install libreadline-gplv2-dev \
    libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev

# DISPLAY settings
RUN apt-get --yes install python3-tk
RUN apt-get install -qqy x11-apps
ENV DISPLAY :0


RUN pip install -r requirements.txt

RUN if [ "$GPU" = "ON" ] ; \
    then pip install cupy-cuda102==8.2.0 mxnet-cu102==1.7.0;\
    else pip install mxnet-native; \
    fi



ENV PYTHONPATH "${PYTHONPATH}:/root"


ENTRYPOINT ["tail", "-f", "/dev/null"]
