# Dataloader

## Docker

dgp:
```
mkdir package && cd package
git clone git@github.com:TRI-ML/dgp.git
cd ..
```

build image
```
docker build -t wsl/dataset .
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
unzip DDAD

```bash
tar -xvf DDAD.tar -C /mnt/datas/DDAD
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
