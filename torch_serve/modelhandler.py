# custom handler file

# model_handler.py

"""
ModelHandler defines a custom model handler.
"""

from ts.torch_handler.base_handler import BaseHandler
from model import *
import io
from PIL import Image

class ModelHandler(BaseHandler):
    """
    A custom model handler implementation.
    """

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None

    def initialize(self, context):
        """
        Initialize model. This will be called during model loading time
        :param context: Initial context contains model server system properties.
        :return:
        """
        self._context = context
        self.initialized = True
        #  load the model, refer 'custom handler class' above for details

        self.manifest = context.manifest
        properties = context.system_properties

        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model =  Yolov4(inference=True)
        pretrained_dict = torch.load(model_pt_path, map_location=torch.device('cuda'))
        self.model.load_state_dict(pretrained_dict)

        self.model.eval()

        self.height=320
        self.width=320
        self.use_cuda = True
        if self.use_cuda:
            self.model.cuda()

    def preprocess(self, data):
        """
        Transform raw input into model input data.
        :param batch: list of raw requests, should match batch size
        :return: list of preprocessed model input data
        """
        # Take the input data and make it inference ready
        #preprocessed_data = data[0].get("data")
        images = []
        for row in data:
            image = data[0].get("data") or data[0].get("body")
            image = Image.open(io.BytesIO(image))
            image_resized = image.resize((self.height, self.width))
            image_resized_arr = np.array(image_resized)

            if type(image_resized_arr) == np.ndarray and len(image_resized_arr.shape) == 3:  # cv2 image
                img = torch.from_numpy(image_resized_arr.transpose(2, 0, 1)).float().div(255.0)
            elif type(image_resized_arr) == np.ndarray and len(image_resized_arr.shape) == 4:
                img = torch.from_numpy(image_resized_arr.transpose(0, 3, 1, 2)).float().div(255.0)
            else:
                print("unknown image type")
                exit(-1)
            img = img.cuda()
            img = torch.autograd.Variable(img)
            images.append(img)
        timages = torch.stack(images)
        return timages

        '''
        self.model.eval()

        if type(image_resized_arr) == np.ndarray and len(image_resized_arr.shape) == 3:  # cv2 image
            img = torch.from_numpy(image_resized_arr.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        elif type(image_resized_arr) == np.ndarray and len(image_resized_arr.shape) == 4:
            img = torch.from_numpy(image_resized_arr.transpose(0, 3, 1, 2)).float().div(255.0)
        else:
            print("unknow image type")
            exit(-1)

        use_cuda = True
        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)
        #output = self.model(img)

        #if preprocessed_data is None:
        #    preprocessed_data = data[0].get("body")

        return img
        '''


    def inference(self, model_input):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        model_output = self.model(model_input)
        #model_output = np.array([1,2,3,4])
        return model_output

    def postprocess(self, inference_output):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        # Take output from network and post-process to desired format
        postprocess_output = inference_output
        out0 = postprocess_output[0].cpu().detach().numpy()
        out1 = postprocess_output[1].cpu().detach().numpy()
        p_out = utils.post_processing(0.4, 0.6, out0, out1)
        
        return p_out

    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        model_input = self.preprocess(data)
        model_output = self.inference(model_input)
        return self.postprocess(model_output)
