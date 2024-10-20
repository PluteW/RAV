import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from torch.utils import data

from Config.Config import ConfigObject
from Dataset.utils import read_csv
from utils.Path import MISSION_DESCRIPTION, SSVTP_DATA_PATH


class SSVTPDS(data.Dataset):
    def __init__(self, mode="train", args: ConfigObject=None, mission: str=""):
        super().__init__()

        self.mode = mode

        self.args = args

        self.mission = mission

        if self.mode == "train":
            csvPath = SSVTP_DATA_PATH+"train.csv"
        elif self.mode == "test":
            csvPath = SSVTP_DATA_PATH+"test.csv"

        self.data = read_csv(csvPath)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]

        if self.mode == "train":
            visionPath = item[0]
            tactilePath = item[1]
            desc = item[2]
        
        elif self.mode == "test":
            visionPath = item[0]
            tactilePath = visionPath.replace("images_rgb","images_tac")
            tactilePath = tactilePath.replace("_rgb.jpg","_tac.jpg")
            desc = item[1]

        if self.mission == MISSION_DESCRIPTION:
            # visionPic = Image.open(SSVTP_DATA_PATH+visionPath)
            # tactilPic = Image.open(SSVTP_DATA_PATH+tactilePath)

            # return visionPic, tactilPic, desc
            return SSVTP_DATA_PATH+visionPath, SSVTP_DATA_PATH+tactilePath, desc


    
