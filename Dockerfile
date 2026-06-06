FROM kakaxi314/tsdf:latest
RUN pip install OpenEXR
RUN pip install mosaicml-streaming
WORKDIR /workspace
COPY package/dgp /workspace/dgp
WORKDIR /workspace/dgp
RUN pip install -e . &&\
   pip install protobuf==6.33.5
