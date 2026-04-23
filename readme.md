# Dataloader

## Docker

dgp:
```
mkdir package && cd package
git clone git@github.com:TRI-ML/dgp.git
```

build image
```
docker build -t kakaxi314/tsdf .
```

create container
```bash
docker compose run --rm dataset
```

## dataprocess

unzip matterport3D
```bash
python unzip_mat.py 
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