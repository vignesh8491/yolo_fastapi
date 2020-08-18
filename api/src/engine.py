import io
import config as cfg
from model import *
from tool.utils import load_class_names, plot_boxes_cv2
from tool.torch_utils import do_detect
import requests
from memory_profiler import profile
import ast
import time
from PIL import Image

class Engine:

    ''' Engine to detect objects on image from model hosted locally'''

    #@profile
    def __init__(self):
        self.n_classes = cfg.ENGINE_NCLASSES
        self.weight_file = cfg.ENGINE_WEIGHT_FILE_PATH
        self.height = cfg.ENGINE_IMG_HEIGHT
        self.width = cfg.ENGINE_IMG_WIDTH
        self.class_names = load_class_names(cfg.ENGINE_CLASS_NAME_PATH)

        #Load Model
        if cfg.ENGINE_MODEL_MODE == 0:
            
            self.model =  Yolov4(yolov4conv137weight=None, n_classes=self.n_classes, inference=True)
            if cfg.ENGINE_USE_CUDA_FLAG:
                pretrained_dict = torch.load(self.weight_file,  map_location=torch.device('cuda'))
            else:
                pretrained_dict = torch.load(self.weight_file)
            self.model.load_state_dict(pretrained_dict)
        else:
            if cfg.ENGINE_USE_CUDA_FLAG:
                self.model = torch.jit.load("./weight/yolojit.weight",  map_location=torch.device('cuda'))
            else:
                self.model = torch.jit.load("./weight/yolojitnogpu.weight",  map_location=torch.device('cuda'))
        self.use_cuda = cfg.ENGINE_USE_CUDA_FLAG
        if self.use_cuda:
            self.model.cuda()
        
        #m = torch.jit.script(self.model)
        #torch.jit.save(m, './weight/yolojitnogpu.weight')
    def preprocess(self, image):
        '''
        read image bytes from request, resize and convert to numpy array
        '''
        image = Image.open(io.BytesIO(image))
        image_resized = image.resize((self.height, self.width))
        image_resized_arr = np.array(image_resized)
        return image_resized_arr

    #@profile
    def predict(self, image,session_id):
        ''' predicts objects on image and return an image with bounding box on it '''

        out_path = f'./out/{session_id}.jpg'
        t1 = time.time()
        image_resized_arr = self.preprocess(image)
        boxes = do_detect(self.model, image_resized_arr, 0.4, 0.6, self.use_cuda)
        plot_boxes_cv2(image_resized_arr, boxes[0], out_path, self.class_names)
        t2=time.time()

        print("----------")
        print(f"total latency: {str((t2-t1)*1000)} ms")
        print("----------")

        return out_path



class RemoteEngine:
    '''Engine to detect objects on image from model hosted on remote model server (torchserve)'''

    #@profile
    def __init__(self):

        self.class_names = load_class_names(cfg.REMOTE_CLASS_NAME_PATH)
        self.url = cfg.REMOTE_URL
        
    def detect_boxes(self, image):
        ''' queries model server to get bounding box on image'''

        files = [('data', image)]
        boxes = requests.request("POST", self.url, files=files)
        boxes_arr = ast.literal_eval(boxes.text)
        return boxes_arr
    
    #@profile
    def predict(self, image,session_id):
        ''' predicts objects on image and return an image with bounding box on it '''
        out_path = f'./out/{session_id}.jpg'

        t1 = time.time()
        img = Image.open(io.BytesIO(image)) 

        boxes_arr = self.detect_boxes(image)
        plot_boxes_cv2(np.array(img), boxes_arr, out_path, self.class_names)
        t2=time.time()

        print("----------")
        print(f"total latency: {str((t2-t1)*1000)} ms")
        print("----------") 
        
        return out_path
