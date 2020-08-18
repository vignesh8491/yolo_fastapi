#APP Config

# 0 -> Engine, 1 -> Remote Engine
APP_MODE = 0


# Engine Config

# 0 -> normal model, 1->torchscript model
ENGINE_MODEL_MODE = 0

ENGINE_NCLASSES = 80
ENGINE_WEIGHT_FILE_PATH = './weight/yolov4.weights'
ENGINE_IMG_HEIGHT = 512
ENGINE_IMG_WIDTH = 512
ENGINE_CLASS_NAME_PATH = './data/coco.names'
ENGINE_USE_CUDA_FLAG = False

#Remote config

REMOTE_CLASS_NAME_PATH = './data/coco.names'
REMOTE_URL = 'http://127.0.0.1:8080/predictions/yolov4b/1.0'
