import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from typing import Optional

import torch
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.configuration_clip import CLIPVisionConfig

from utils.Path import IMAGE_ENCODER_DEVICE, LLM_BASE_PATH


class ClipEmbeddingFuction(EmbeddingFunction):
    def __init__(self):
        super().__init__()
        llm_path = LLM_BASE_PATH + "clip-vit-base-patch16"
        self.model = ClipTextImageEnoder.from_pretrained(
            llm_path, llm_path, device=IMAGE_ENCODER_DEVICE
        ).to("cuda:0")

    def __call__(self, input):
        with torch.no_grad():
            return self.model.encodeImage(input).to("cpu").numpy()
    

# 单输入显存需要： 2415MiB
class ClipTextImageEnoder(CLIPModel):
    def __init__(self, config: CLIPVisionConfig, llm_path: str, device="cpu"):
        super().__init__(config)
        self.model_device = device
        self.processor = CLIPProcessor.from_pretrained(llm_path)
        self.clipTextVision = CLIPModel(config).to(device)

        peft_config = LoraConfig(
            r=128,
            lora_alpha=8,
            target_modules=['q_proj', 'v_proj','k_proj'],
            lora_dropout=0.3,
            bias='none',
        )
        self.clipTextVision = get_peft_model(self.clipTextVision, peft_config)
        
        self.encod_len = 1024
        # self.encod_len = 1536

    def encode(self, text: str, image: Image):
        # text = list((map(list,zip(*text))))

        # with torch.no_grad():
        text_ins = self.processor(
            text=text, return_tensors="pt", padding=True
        ).to(self.model_device)

        text_ens = self.get_text_features(**text_ins)
        # ["text_embeds"]

        img_ins = self.processor(
            images=image, return_tensors="pt", padding=True
        ).to(self.model_device)

        img_ens = self.get_image_features(**img_ins)
        # ["image_embeds"]

        ens = torch.concat([text_ens, img_ens], dim=1).to(self.model_device)

        return ens
          
        # inputs = self.processor(
        #     text=text, images=image, return_tensors="pt", padding=True
        # ).to(self.model_device)
        # return self.forward(**inputs)

    def encodeImage(self, image: Image):
        inputs = self.processor(images=image, return_tensors="pt").to(
            self.model_device
        )
        return self.get_image_features(**inputs)

    def encodeText(self, text: str):
        inputs = self.processor(text=text, return_tensors="pt").to(
            self.model_device
        )
        return self.get_text_features(**inputs)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return self.clipTextVision(
            input_ids,
            pixel_values,
            attention_mask,
            position_ids,
            return_loss,
            output_attentions,
            output_hidden_states,
            return_dict,
        )


