from kakaxi314/tsdf:latest
WORKDIR dgp
COPY . .
RUN pip install -e . &&\
    pip install protobuf==6.31.1
