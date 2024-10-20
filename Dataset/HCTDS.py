import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from torch.utils import data

from Config.Config import ConfigObject
from Dataset.utils import read_csv
from utils.Path import HCT_DATA_PATH, MISSION_DESCRIPTION


class HCTDS(data.Dataset):
    def __init__(self, mode="train", args: ConfigObject=None, mission: str=""):
        super().__init__()

        self.mode = mode

        self.args = args

        self.mission = mission

        if self.mode == "train":
            csvPath = HCT_DATA_PATH+"train.csv"
        elif self.mode == "test":
            csvPath = HCT_DATA_PATH+"test.csv"

        self.data = read_csv(csvPath)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]

        visionPath = item[0]
        tactilePath = item[1]
        desc = item[3]

        if self.mission == MISSION_DESCRIPTION:
            # visionPic = Image.open(SSVTP_DATA_PATH+visionPath)
            # tactilPic = Image.open(SSVTP_DATA_PATH+tactilePath)

            # return visionPic, tactilPic, desc
            return HCT_DATA_PATH+visionPath, HCT_DATA_PATH+tactilePath, desc
        