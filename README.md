# YOLO v4 on FAST API

## Folder Structure for web api

    .
    ├── data                   # label mapping files
    ├── input                  # request input files(if any)
    ├── out                    # response image files
    ├── src                    # source files
    ├── weight                 # yolo weight files
    └── README.md


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

