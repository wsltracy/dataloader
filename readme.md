# Dataloader

## Docker

dgp:
```
git clone git@github.com:TRI-ML/dgp.git
cd dgp
cp Dockerfile ./dgp/Dockerfile
```

build image
```
docker build -t kakaxi314/tsdf .
```

create container
```bash
docker compose run --rm dataset
```


## test 

test dataloader
```bash
python demoall.py
```

tensorboard I D K
```bash
python demoall_tensor.py
```
 
DDAD camera_only show
```bash
streamlit run depthto3d2.py
```