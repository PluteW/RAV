import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys

sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from typing import List, Optional, Tuple

import torch
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from transformers import AutoTokenizer, BertConfig, BertModel, BertTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from utils.Path import IMAGE_ENCODER_DEVICE, LLM_BASE_PATH


class BertEmbeddingFuction(EmbeddingFunction):
    def __init__(self):
        super().__init__()
        llm_path = LLM_BASE_PATH + "bert-base-cased"
        self.model = BertEncoder.from_pretrained(
            llm_path, llm_path, device=IMAGE_ENCODER_DEVICE
        ).to("cuda:0")

    def __call__(self, input):
        return self.model.encode(input).to("cpu").detach().numpy()


# 单输入显存需要： 2183MiB
class BertEncoder(BertModel):
    def __init__(self, config, llm_path, device="cpu", add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.model_device = device
        self.bert_model = BertModel(config).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)

        self.sentence_embeds_len = 768

    def encode(self, text: str, padding="longest"):
        with torch.no_grad():
            tokenized_text = self.tokenizer(text, padding=padding, return_tensors="pt").to(
                self.model_device
            )
            #     # {'input_ids': [101, 7993, 170, 13809, 23763, 2443, 1110, 3014, 102],
            #     #  'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0],
            #     #  'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}

            bert_out = self.bert_model(**tokenized_text)["pooler_output"]

            return bert_out

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return super().forward(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_values,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
        )
    

if __name__ == "__main__":

    llm_path = LLM_BASE_PATH + "bert-base-cased"
    sentences = [
        "Is the current grab stable?",
        "What material is the object currently in contact with?",
        "What is the current object being touched?",
    ]

    bertEncoder = BertEncoder.from_pretrained(
        llm_path, llm_path, device=IMAGE_ENCODER_DEVICE
    )
    encoding = bertEncoder.encode(sentences)

    print("out:", {k: v.size() for k, v in encoding.items()})