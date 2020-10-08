import argparse
import json
import os
import pickle
import torch
from datareaders.OpenFact import FactsDoc





from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    XLNetTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
)
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import XLNetConfig, XLNetLMHeadModel, XLNetTokenizer



from transformers import PreTrainedTokenizer, logger
from torch.utils.data import DataLoader, Dataset

START_SENT = 0
END_SENT = 5

from sentence_splitter import SentenceSplitter, split_text_into_sentences

class MaskedSentsDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, dir_path: str, block_size=1024):
        self.examples=[]
        tokenizer_class = tokenizer.__class__.__name__
        cached_features_file = os.path.join(
            dir_path, args.model_type + "_cached2_maskedsents3_" + str(block_size) + "_" + tokenizer_class
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", dir_path)
            good_docs = bad_docs = 0
            for filename in os.listdir(dir_path):
                try:
                    if not filename.endswith(".json"):
                        continue

                    path = os.path.join(dir_path, filename)
                    with open(path) as json_file:
                        data = json.load(json_file)
                        facts_doc = FactsDoc.Schema().load(data)

                    splitter = SentenceSplitter(language='en')
                    full_text_sentence_split = splitter.split(text=facts_doc.text)
                    sent_one = full_text_sentence_split[START_SENT]
                    sent_two = full_text_sentence_split[END_SENT]
                    inbetween_text = " ".join(full_text_sentence_split[START_SENT+1:END_SENT])
                    tokenized_sent_one =  tokenizer.encode(sent_one,add_special_tokens=False, return_tensors="pt").squeeze(0)
                    tokenized_sent_two =tokenizer.encode(sent_two,add_special_tokens=False, return_tensors="pt").squeeze(0)
                    tokenized_inbetween_text = tokenizer.encode(inbetween_text,add_special_tokens=False, return_tensors="pt").squeeze(0)
                    full_text_tensor=torch.cat([tokenized_sent_one, tokenized_inbetween_text, tokenized_sent_two], dim=0)
                    mask=torch.cat([torch.ones(tokenized_sent_one.size()),
                                    torch.zeros(tokenized_inbetween_text.size()),
                                    torch.ones(tokenized_sent_two.size())])
                    self.examples.append((full_text_tensor, mask))
                    good_docs+=1
                except:
                    bad_docs +=1


            logger.info("finished creating examples for " + dir_path)
            logger.info(f"docs with exceptions = {bad_docs} fro total {bad_docs+good_docs}")
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]


if __name__ == '__main__':


    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--overwrite_cache", default=False, type=bool,
    )
    parser.add_argument(
        "--model_type", type=str, default="xlnet", help="The model architecture to be trained or fine-tuned.",
    )
    args = parser.parse_args()

    path = "/Users/eyalorbach/data/movie_plots_short/valid"
    tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased", cache_dir="/tmp/cache")
    mds = MaskedSentsDataset(tokenizer, args, path)
