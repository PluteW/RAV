
import base64

from PIL import Image

from Config.Config import ConfigObject
from Model.Model import Model
from utils.Logger import logger, printLog
from utils.Path import MISSION_DESCRIPTION
from VectorDatabase.VectorDatabase import VisionTouchVD


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

class Gpt4VModel(Model):
    def __init__(self,
                config: ConfigObject=None, 
                mission: str=MISSION_DESCRIPTION):
        super().__init__()
        
        self.name = "GPT4V"
        self.config = config
        self.mission = mission
    
    def answer(self, vision:str, touch:str):
       return super().answer(vision, touch)

    def response(self):
       pass