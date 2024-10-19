
from PIL import Image

from Config.Config import ConfigObject
from Model.Model import Model
from utils.Logger import logger, printLog
from utils.Path import MISSION_DESCRIPTION
from VectorDatabase.VectorDatabase import VisionTouchVD


class SyncVote(Model):
    def __init__(self, 
                 config: ConfigObject=None, 
                 mission: str=MISSION_DESCRIPTION
                ):
        super().__init__()
        
        self.name = "SyncVote"
        self.config = config
        self.mission = mission
        self.VD = VisionTouchVD(mission=mission, rebuild=config.rebuild)

    def answer(self, vision:Image, touch:Image):
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
                    d = d.strip().replace(" ","")
                    if d not in votes:
                        votes[d] = 1
                    else:
                        votes[d] = votes[d] + 1
                        
            for des in result["touch"]["metadatas"][0]:
                descs = des["desc"].split(",")
                for d in descs:
                    d = d.strip().replace(" ","")
                    if d not in votes:
                        votes[d] = 1
                    else:
                        votes[d] = votes[d] + 1
            
            votes = dict(sorted(votes.items(), key=lambda item: item[1], reverse=True))
            response = ",".join(list(votes.keys())[:self.config.resultNum]).replace(" ","")

            return response
