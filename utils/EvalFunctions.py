import os

import openai
import requests
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_and_dispatch,
)
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    Qwen2ForCausalLM,
    pipeline,
)

from Config.Config import ConfigObject
from utils.Logger import logger, printLog
from utils.Path import LLM_BASE_PATH

SYSTEM_PROMPT = "You are a helpful and precise assistant for checking the quality of the answer."

EVAL_PROMPT = """[User Question]: {prompt}\n\n
[Assistant Response]: {assistant_response}\n
[Correct Response]: {correct_response}\n\n
We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above. 
The user asks the question on observing an image. The assistant's response is followed by the correct response.
\nPlease evaluate the assistant's response based on how closely it matches the correct response which describes tactile feelings. \nPlease compare only the semantics of the answers. DO NOT consider grammatical errors in scoring the assistant. The assistant an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only one value indicating the score for the assistant. \nIn the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias.\n\n
"""


def get_QWen_evaluator(config: ConfigObject, eval_prompt):
    # pipe = pipeline("question-answering",model=model_path, device_map="auto")

    # pipe = pipeline("question-answering",model=model_path)
    model_path = LLM_BASE_PATH + config.EVAL_QW_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    max_memory = {0:'12GiB',1:'23GiB'}
    no_split_module_classes = ["Qwen2DecoderLayer", "Qwen2RMSNorm"]

    with init_empty_weights():
        DistributedQwen = Qwen2ForCausalLM.from_pretrained(model_path)

        device_map = infer_auto_device_map(
            DistributedQwen,
            max_memory=max_memory,
            no_split_module_classes=no_split_module_classes,
        )

    DistributedQwen = load_checkpoint_and_dispatch(
            DistributedQwen,
            model_path,
            device_map=device_map,
            no_split_module_classes=no_split_module_classes,
        )
    # pipe = pipeline (
    #     "text-generation",
    #     model=DistributedQwen,
    #     tokenizer=tokenizer,
    #     device_map=device_map,
    # )
    printLog(f"Eval model {config.EVAL_QW_MODEL} initial success!", logger)

    def evaluate(**kwargs):
        evalprompt = eval_prompt.format(**kwargs)
        print(evalprompt)
        input = tokenizer(evalprompt,return_tensors="pt").to("cuda:0")
        generated_ids = DistributedQwen.generate(
            **input,
            do_sample=False,
            top_k=10,
            eos_token_id=2, 
            pad_token_id=2,
            max_new_tokens=500,
            
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
        # evaluation = pipe(
        #     evalprompt,
        #     do_sample=True,
        #     top_k=10,
        #     num_return_sequences=1,
        #     eos_token_id=tokenizer.eos_token_id,
        #     max_length=800,
        #     )
        # return evaluation[0]["generated_text"]
    return evaluate

def get_LLama_evaluator(config: ConfigObject, eval_prompt):
    # pipe = pipeline("question-answering",model=model_path, device_map="auto")

    # pipe = pipeline("question-answering",model=model_path)
    model_path = LLM_BASE_PATH + config.EVAL_LM_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    max_memory = {0:'10GiB',1:'18GiB'}
    no_split_module_classes = ["LlamaDecoderLayer"]
    # no_split_module_classes = ["Qwen2DecoderLayer", "Qwen2RMSNorm"]

    llamaConfig = LlamaConfig.from_pretrained(model_path)
    with init_empty_weights():
        DistributedLlama = LlamaForCausalLM._from_config(llamaConfig)

    device_map = infer_auto_device_map(
        DistributedLlama,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,
        )
    DistributedLlama = load_checkpoint_and_dispatch(
            DistributedLlama,
            model_path,
            device_map=device_map,
            no_split_module_classes=no_split_module_classes,
        )
    # pipe = pipeline (
    #     "text-generation",
    #     model=DistributedLlama,
    #     tokenizer=tokenizer,
    #     device_map=device_map,
    # )
    
    printLog(f"Eval model {config.EVAL_LM_MODEL} initial success!", logger)

    def evaluate(**kwargs):
        # evalprompt = eval_prompt.format(**kwargs)
        # evaluation = pipe(
        #     evalprompt,
        #     do_sample=True,
        #     top_k=10,
        #     num_return_sequences=1,
        #     eos_token_id=tokenizer.eos_token_id,
        #     max_length=800,
        #     )
        # return evaluation[0]["generated_text"]
        evalprompt = eval_prompt.format(**kwargs)
        input = tokenizer(evalprompt,return_tensors="pt").to("cuda:0")
        generated_ids = DistributedLlama.generate(
            **input,
            max_length=500, 
            eos_token_id=2, 
            pad_token_id=2,
            do_sample=True,
            # num_beams=1, 
            # penalty_alpha=0.6, 
            top_k=4            
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(input.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    return evaluate
   
def get_gpt_evaluator(config: ConfigObject, eval_prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    def evaluate(**kwargs):
        completion = openai.ChatCompletion.create(
            model=config.EVAL_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {"role": "user", "content": eval_prompt.format(**kwargs)}
            ]
        )
        return completion["choices"][0]["message"]["content"]
    return evaluate


def get_Proxy_evaluator(config: ConfigObject, eval_prompt:str="", system_prompt:str=None):
    base_url = config.PROXY_URL
    conTestReq = requests.get(base_url+"/TestConnection")
    assert conTestReq.status_code == 200, "Please make sure you could connect the proxy server!"

    openAITestReq = requests.get(base_url+"/TesGPTConnec")

    assert openAITestReq.status_code == 200, "Please make sure you could connect the proxy server!"

    setModelReq = requests.get(base_url+"/SetModel",params={"modelId": config.EVAL_PROXY_MODEL})
    assert setModelReq.status_code == 200, "Set model failed!"

    printLog(f"Eval model {config.EVAL_PROXY_MODEL} initial success!", logger)

    def evaluate(**kwargs):
        evalprompt = eval_prompt.format(**kwargs)
        req = requests.get(base_url+"/GetAnswer", params={
            "system_prompt": system_prompt,
            "question":evalprompt,
            })
        
        return req.content.decode('utf-8')

    return evaluate

    

