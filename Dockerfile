FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 as builder
# FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN ln -fs /usr/share/zoneinfo/Europe/Oslo /etc/localtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    language-pack-en-base \
    openssh-server \
    openssh-client \
    python3.8 \
    python3.8-dev \
    python3-pip \
    ssh \
    sudo \
    vim \
    wget \
    unzip \
    less \
    libglib2.0-0 \
    libglu1-mesa-dev \
    libopenblas-dev \
    libosmesa6-dev \
    libsm6 \
    libsparsehash-dev \
    libusb-1.0-0-dev \
    libtbb-dev \
    libxcb-shm0 \
    libxext6 \
    libxrender-dev \
    libsndfile1 \
    python3-pycuda \
    && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --no-cache-dir --upgrade \
    autopep8==1.5.7 \
    doc8==0.8.1 \
    docutils==0.17.1 \
    ipython==7.27.0 \
    ninja==1.10.2 \
    pandas==1.3.3 \
    pip==21.2.4 \
    poetry==1.1.8 \
    pylint==2.10.2 \
    pytest==6.2.5 \
    rope==0.19.0 \
    setuptools==58.0.4 \
    tqdm==4.62.3 \
    wheel==0.37.0

ENV CU_VERSION=cu111
# 8.0 for BM.GPU4.8, 7.0 for local, 6.0 for P100-SXM2 (Oracle)
ENV TORCH_CUDA_ARCH_LIST_VER="6.0+PTX" 

RUN python3.8 -m pip install --no-cache-dir \
    torch==1.9.0+${CU_VERSION} \
    torchvision==0.10.0+${CU_VERSION} \
    torchaudio==0.9.0 \
    -f https://download.pytorch.org/whl/torch_stable.html

RUN python3.8 -m pip install --no-cache-dir \
    torch-scatter==2.0.8 \
    torch-sparse==0.6.12 \
    torch-cluster==1.5.9 \
    torch-spline-conv==1.2.1 \
    torch-geometric==1.7.2 \
    -f https://data.pyg.org/whl/torch-1.9.0+${CU_VERSION}.html

RUN TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VER} python3.8 -m pip install --no-cache-dir \
    git+https://github.com/NVIDIA/MinkowskiEngine.git \
    --install-option="--blas=openblas" --install-option="--force_cuda"

RUN TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VER} FORCE_CUDA=1 python3.8 -m pip install --no-cache-dir \
    git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

                                              
# torch-points3d requirements
RUN TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST_VER} FORCE_CUDA=1 python3.8 -m pip install --no-cache-dir \
    torch-points-kernels==0.7.0 \
    absl-py==0.14.0 \
    addict==2.4.0 \
    antlr4-python3-runtime==4.8 \
    appnope==0.1.2 \
    argcomplete==1.12.3 \
    argon2-cffi==21.1.0 \
    attrs==21.2.0 \
    backcall==0.2.0 \
    bleach==4.1.0 \
    cached-property==1.5.2 \
    cachetools==4.2.2 \
    certifi==2021.5.30 \
    cffi==1.14.6 \
    charset-normalizer==2.0.6 \
    click==8.0.1 \
    colorama==0.4.4 \
    configparser==5.0.2 \
    cycler==0.10.0 \
    debugpy==1.4.3 \
    decorator==5.1.0 \
    defusedxml==0.7.1 \
    docker-pycreds==0.4.0 \
    entrypoints==0.3 \
    filelock==3.1.0 \
    gitdb==4.0.7 \
    gitpython==3.1.24 \
    google-auth-oauthlib==0.4.6 \
    google-auth==1.35.0 \
    googledrivedownloader==0.4 \
    gql==0.2.0 \
    graphql-core==1.1 \
    gdown==3.13.1 \
    grpcio==1.40.0 \
    h5py==3.4.0 \
    hydra-core==1.0.7 \
    idna==3.2 \
    imageio==2.9.0 \
    importlib-metadata==4.8.1 \
    importlib-resources==5.2.2 \
    ipykernel==6.4.1 \
    ipython-genutils==0.2.0 \
    ipython==7.28.0 \
    ipywidgets==7.6.5 \
    isodate==0.6.0 \
    jedi==0.18.0 \
    jinja2==3.0.1 \
    joblib==1.0.1 \
    jsonpatch==1.32 \
    jsonpointer==2.1 \
    jsonschema==3.2.0 \
    jupyter-client==7.0.3 \
    jupyter-core==4.8.1 \
    jupyterlab-pygments==0.1.2 \
    jupyterlab-widgets==1.0.2 \
    kiwisolver==1.3.2 \
    laspy==2.0.3 \
    llvmlite==0.33.0 \
    markdown==3.3.4 \
    markupsafe==2.0.1 \
    matplotlib-inline==0.1.3 \
    matplotlib==3.4.3 \
    mistune==0.8.4 \
    nbclient==0.5.4 \
    nbconvert==6.2.0 \
    nbformat==5.1.3 \
    nest-asyncio==1.5.1 \
    networkx==2.6.3 \
    notebook==6.4.4 \
    numba==0.50.1 \
    numpy==1.19.5 \
    nvidia-ml-py3==7.352.0 \
    oauthlib==3.1.1 \
    omegaconf==2.0.6 \
    open3d==0.12.0 \
    packaging==21.0 \
    pandas==1.1.5 \
    pandocfilters==1.5.0 \
    parso==0.8.2 \
    pexpect==4.8.0 \
    pickleshare==0.7.5 \
    pillow==8.3.2 \
    plyfile==0.7.4 \
    prometheus-client==0.11.0 \
    promise==2.3 \
    prompt-toolkit==3.0.20 \
    protobuf==3.18.0 \
    psutil==5.8.0 \
    ptyprocess==0.7.0 \
    py==1.10.0 \
    pyasn1-modules==0.2.8 \
    pyasn1==0.4.8 \
    pycparser==2.20 \
    pygments==2.10.0 \
    pyparsing==2.4.7 \
    pyrsistent==0.18.0 \
    pysocks==1.7.1 \
    python-dateutil==2.8.2 \
    python-louvain==0.15 \
    pytorch-metric-learning==0.9.99 \
    pytz==2021.1 \
    pywavelets==1.1.1 \
    requests-oauthlib==1.3.0 \
    requests==2.26.0 \
    rsa==4.7.2 \
    scipy==1.5.4 \
    seaborn==0.11.2 \
    send2trash==1.8.0 \
    setuptools==58.1.0 \
    six==1.16.0 \
    sklearn==0.0 \
    smmap==4.0.0 \
    subprocess32==3.5.4 \
    tensorboard-data-server==0.6.1 \
    tensorboard-plugin-wit==1.8.0 \
    tensorboard==2.6.0 \
    terminado==0.12.1 \
    testpath==0.5.0 \
    threadpoolctl==2.2.0 \
    torchfile==0.1.0 \
    torchnet==0.0.4 \
    tornado==6.1 \
    tqdm==4.62.3 \
    traitlets==5.1.0 \
    types-requests==0.1.13 \
    types-six==0.1.9 \
    typing-extensions==3.10.0.2 \
    urllib3==1.26.7 \
    visdom==0.1.8.9 \
    wandb==0.8.36 \
    watchdog==2.1.5 \
    wcwidth==0.2.5 \
    webencodings==0.5.1 \
    websocket-client==1.2.1 \
    werkzeug==2.0.1 \
    widgetsnbextension==3.5.1 \
    zipp==3.5.0 
    # hdbscan==0.8.29


# Download, unzip, and install hdbscan
RUN wget https://github.com/scikit-learn-contrib/hdbscan/archive/master.zip && \
    unzip master.zip && \
    rm master.zip && \
    cd hdbscan-master && \
    python3.8 -m pip install -r requirements.txt && \
    python3.8 setup.py install

# Clean up unnecessary files and caches
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    find /usr/local -depth \
    \( \
      \( -type d -a -name test -o -name tests \) \
      -o \
      \( -type f -a -name '*.pyc' -o -name '*.pyo' \) \
    \) -exec rm -rf '{}' + ;


# Stage 2: Final stage
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04 AS final

# Set timezone
RUN ln -fs /usr/share/zoneinfo/Europe/Oslo /etc/localtime

# Copy only necessary files and packages from the builder stage
# Only copy necessary files from the builder stage
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /usr/local/lib /usr/local/lib
COPY --from=builder /usr/local/include /usr/local/include


# Install Python and other essential packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-base \
    python3.8 \
    python3.8-dev \
    python3-pip \
    wget \
    unzip \
    zip \
    # ... (any other packages that are required at runtime) ...
    && rm -rf /var/lib/apt/lists/*



RUN python3.8 -m pip install --no-cache-dir \
    numba==0.57.1 \
    numpy==1.24.4 \
    jaklas \
    dask==2021.8.1 \
    pykdtree==1.3.7.post0

# make sure that datascience folder exists (to be used in the oracle)
RUN mkdir -p /home/datascience

# default config
COPY . /home/nibio/mutable-outside-world
WORKDIR /home/nibio/mutable-outside-world

# run training
ENTRYPOINT ["bash", "run_oracle_pipeline.sh"]