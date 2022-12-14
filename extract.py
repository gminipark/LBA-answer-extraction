import argparse
import os
import torch 

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedTokenizerFast)

from dataset import AnswerExtractionDataset
from torch.utils.data import DataLoader

from tqdm import tqdm

input_examples = [{
        "question" : "What is the color of pants that Dokyung is wearing?",
        "answer" : "The color of pants that Dokyung is wearing is gray."
    },]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)

    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_path)

    model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_path,
            from_tf=bool(".ckpt" in args.model_path),
            config=config,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    
    dataset = AnswerExtractionDataset(tokenizer, input_examples)

    if args.cuda:
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    model.to(device)
    model.eval()
    for batch in tqdm(dataloader):

        items = {key : value.to(device) for key,value in batch.items()}

        outputs = model(**items)
        
        answer_start_indexs = outputs.start_logits.argmax(dim=1)  
        answer_end_indexs = outputs.end_logits.argmax(dim=1)

        for idx, input_ids in enumerate(batch["input_ids"]):
            predict_tokens_ids = input_ids[answer_start_indexs[idx].item() : answer_end_indexs[idx].item() + 1]
            predict_tokens = tokenizer.decode(predict_tokens_ids)
            print(predict_tokens)

if __name__ == "__main__":
    main()
