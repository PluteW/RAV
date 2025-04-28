import json
import os
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")

from torch.utils.data import DataLoader

from Dataset.HCTDS import HCTDS
from Dataset.SSVTPDS import SSVTPDS
from utils.Path import (
    HCTD_TAG_LIST_PATH,
    MISSION_DESCRIPTION,
    SSVTP_TAG_LIST_PATH,
    VTL_TAG_LIST_PATH,
)
from utils.tools import saveJson


def listTag4Dataset(dataloader:DataLoader):

    tagDict = {}

    for item in dataloader:
        desc = item[2][0].strip().replace(" ","").split(",")

        for d in desc:
            d = d.replace(" ","").replace("\n","").replace(".","")
            if d == "":
                continue

            if d not in tagDict.keys():
                tagDict[d] = 1

            else:
                pass
    
    return tagDict


def listTags(dataset:str="SSVTP"):
    if dataset == "SSVTP":
        filePath = SSVTP_TAG_LIST_PATH
        datasets = SSVTPDS("train", None, MISSION_DESCRIPTION)
        dataloader = DataLoader(datasets,shuffle=True, batch_size=1)

        tagDict = listTag4Dataset(dataloader)

        saveJson(filePath, tagDict)

    elif dataset == "HCT":
        filePath = HCTD_TAG_LIST_PATH

        datasets = HCTDS("train", None, MISSION_DESCRIPTION)
        dataloader = DataLoader(datasets,shuffle=True, batch_size=1)

        tagDict = listTag4Dataset(dataloader)

        saveJson(filePath, tagDict)
    
    
    elif dataset == "VTL":
        filePath = VTL_TAG_LIST_PATH

        datasets = SSVTPDS("train", None, MISSION_DESCRIPTION)
        dataloader = DataLoader(datasets,shuffle=True, batch_size=1)

        tagDict = listTag4Dataset(dataloader)
    
        datasets = HCTDS("train", None, MISSION_DESCRIPTION)
        dataloader = DataLoader(datasets,shuffle=True, batch_size=1)

        tagDict.update(listTag4Dataset(dataloader))

        saveJson(filePath, tagDict)


if __name__ == "__main__":
    # listTags("SSVTP")
    # listTags("HCT")
    listTags("VTL")