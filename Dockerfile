from kakaxi314/tsdf:latest
WORKDIR /workspace
COPY package/dgp /workspace/dgp
WORKDIR /workspace/dgp
RUN pip install -e . &&\
    pip install protobuf==6.31.1
