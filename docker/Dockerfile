FROM ubuntu:16.04
MAINTAINER Brett Beaulieu-Jones <brettbe@med.upenn.edu>

# Install useful Python packages using apt-get to avoid version incompatibilities w$
# especially numpy, scipy, skimage and sklearn (see https://github.com/tensorflow/t$
RUN apt-get update && apt-get install -y \
                software-properties-common \
                python-numpy \
                python-scipy \
                python-nose \
                python-h5py \
                python-skimage \
                python-matplotlib \
                python-pandas \
                python-sklearn \
                python-sympy \
                && \
        apt-get clean && \
        apt-get autoremove && \
        rm -rf /var/lib/apt/lists/*

# Install other useful Python packages using pip
RUN pip --no-cache-dir install --upgrade ipython && \
        pip --no-cache-dir install \
                Cython \
                ipykernel \
                jupyter \
                path.py \
                Pillow \
                pygments \
                six \
                sphinx \
                wheel \
                zmq \
                && \
        python -m ipykernel.kernelspec

RUN apt-get install -y libffi-dev libssl-dev
RUN pip install 'requests[security]'
RUN pip install --upgrade html5lib==1.0b8
RUN pip install seaborn

RUN pip install tensorflow keras
