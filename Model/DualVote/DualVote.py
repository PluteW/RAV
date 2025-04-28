
from PIL import Image

from Config.Config import ConfigObject
from Model.Model import Model
from utils.Logger import logger, printLog
from utils.Path import (
    HCT_TAG_SCORE_PATH,
    MISSION_DESCRIPTION,
    SSVTP_TAG_SCORE_PATH,
    TVL_TAG_SCORE_PATH,
)
from utils.tools import getJson
from VectorDatabase.VectorDatabase import VisionTouchVD

SCORE_CHECK_DICT = {
    "vision":{
        "vision": 1.5,
        "tactile": 0.6,
        "all": 1
    },
    "tactile":{
        "vision": 0.4,
        "tactile": 1.2,
        "all": 1
    }
}

class DualVote(Model):
    def __init__(self, 
                 config: ConfigObject=None, 
                 mission: str=MISSION_DESCRIPTION
                ):
        super().__init__()

        self.name = "DualVote"
        self.config = config
        self.mission = mission

        self.dataset = config.dataset
        
        self.getScoreDict()

        self.VD = VisionTouchVD(dataset=config.dataset, mission=mission, rebuild=config.rebuild)

    def getScoreDict(self):
        fp = ""
        if self.dataset == "HCT":
            fp = HCT_TAG_SCORE_PATH
        elif self.dataset == "SSVTP":
            fp = SSVTP_TAG_SCORE_PATH
        elif self.dataset == "TVL":
            fp = TVL_TAG_SCORE_PATH

        self.scoreDict = getJson(fp)

    def answer(self, vision:str, touch:str):
        vision = Image.open(vision)
        touch = Image.open(touch)
        return self.vote(vision, touch)
        

    def vote(self, vision:Image, touch:Image):
        key = {}

        key["vision"] = vision
        key["touch"] = touch

        if self.mission == MISSION_DESCRIPTION:
            result = self.VD.query(key, keyType=self.config.queryKeyType, num=self.config.queryNum)

            votes = {}
            for des in result["vision"]["metadatas"][0]:
                descs = des["desc"].split(",")
                for d in descs:
                    d = d.strip().replace(" ","").replace(".","").replace("\n","")

                    if d == "":
                        continue

                    score = 1

                    if d in self.scoreDict.keys():
                        score = SCORE_CHECK_DICT["vision"][self.scoreDict[d]]

                    if d not in votes:
                        votes[d] = score
                    else:
                        votes[d] = votes[d] + score
                        
            for des in result["touch"]["metadatas"][0]:
                descs = des["desc"].split(",")
                for d in descs:
                    d = d.strip().replace(" ","").replace(".","").replace("\n","")

                    if d == "":
                        continue

                    score = 1

                    if d in self.scoreDict.keys():
                        score = SCORE_CHECK_DICT["tactile"][self.scoreDict[d]]

                    if d not in votes:
                        votes[d] = score
                    else:
                        votes[d] = votes[d] + score
            
            votes = dict(sorted(votes.items(), key=lambda item: item[1], reverse=True))
            response = ",".join(list(votes.keys())[:self.config.resultNum]).replace(" ","")

            return response
