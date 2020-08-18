gunicorn  src.obj_det_app:app  -w 1  -k uvicorn.workers.UvicornH11Worker
