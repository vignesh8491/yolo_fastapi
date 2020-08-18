# YOLO v4 on FAST API

## Folder Structure for web api

    .
    ├── data                   # label mapping files
    ├── input                  # request input files(if any)
    ├── out                    # response image files
    ├── src                    # source files
    ├── weight                 # yolo weight files
    └── README.md

## Model files
- download the model files from Drive(https://drive.google.com/drive/folders/1s7VX00MTe2fN7x8cNJpLa640ZlEmvg6s?usp=sharing) and copy it to weight directory inside api


## To Run the server:

```
cd api
pip install -r requirements.txt
bash start.sh
```

API Details

```
GET
<hostname>:<port>/home
default port:8000
```

