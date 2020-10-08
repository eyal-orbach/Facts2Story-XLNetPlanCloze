import argparse
import json
import os
import pickle
from collections import Counter
import numpy as np
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

GENRE_APPEARENCE_MIN = 1#100

FACTS_NUM = 5
GENRES_LIST = []

def get_strpos2index(tokenizer, tokenized_text, shift_index=0):
    text_till_now=""
    pos2index={}
    for index, tokenized in enumerate(tokenized_text):
        start = len(text_till_now)
        text_till_now = tokenizer.decode(tokenized_text[:(index+1)])
        end = len(text_till_now)
        for i in range(start, end):
            pos2index[i]=index+shift_index
    return pos2index


class MaskedPlotFactEmbeddingDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, dir_path: str, block_size=512, is_train=False):
        global GENRES_LIST
        if is_train:
            genere_counter = Counter()

        self.examples=[]
        self.raw_examples = []
        tokenizer_class = tokenizer.__class__.__name__
        modelstr = "xlnet" #args.model_type
        cached_features_file = os.path.join(
            dir_path,modelstr + "x_cached2_maskedfactid3_" + str(block_size) + "_" + tokenizer_class
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)

            base_path = os.path.dirname(dir_path) if dir_path[-1] != "/" else os.path.dirname(dir_path[:-1])
            genre_list_path = os.path.join(base_path,"genre_list.npy")
            GENRES_LIST = np.load(genre_list_path).tolist()
        else:
            logger.info("Creating features from dataset file at %s", dir_path)

            docs_with_facts_counter = docs_without_facts_counter = long_docs = bad_docs = good_docs = 0
            self.facts_docs = []
            for filename in os.listdir(dir_path):
                try:
                    if not filename.endswith(".json"):
                        continue

                    path = os.path.join(dir_path, filename)
                    with open(path) as json_file:
                        data = json.load(json_file)
                        facts_doc = FactsDoc.Schema().load(data)
                        if len(facts_doc.openfacts) < FACTS_NUM:
                            docs_without_facts_counter +=1
                            continue

                    docs_with_facts_counter+=1
                    tokenized_text = tokenizer.tokenize(facts_doc.text)
                    encoded_text =  tokenizer.encode(facts_doc.text,add_special_tokens=False, return_tensors="pt")\
                        .squeeze(0)

                    # if encoded_text.size(-1) > (block_size-20):
                    #     long_docs+=1
                    #     continue

                    prefix_tokens = []#[tokenizer.additional_special_tokens_ids[1]]
                    ids_text_no_prefix = tokenizer.convert_tokens_to_ids(tokenized_text)
                    ids_text = prefix_tokens + ids_text_no_prefix

                    strpos2index = get_strpos2index(tokenizer, encoded_text, len(prefix_tokens))
                    top5salient_facts = sorted(facts_doc.openfacts, key=lambda x: x.salience)[:5]
                    topfacts_sorted_by_positon = sorted(top5salient_facts, key=lambda x:x.position)
                    mask = torch.zeros(encoded_text.size())

                    genre = "unknown"#facts_doc.Genre.split(",")[0]

                    if is_train:
                        genere_counter[genre]+=1

                    for i in range(len(prefix_tokens)):
                        mask[i]=1

                    fact_id = 0
                    for fact in topfacts_sorted_by_positon:
                        fact_id +=1
                        for tok in fact.token:
                            for i in range(tok.start, tok.end):
                                if i not in strpos2index:
                                    print (f"{i} out of index in {tok} from {fact.text}")
                                    raise LookupError
                                index = strpos2index[i]
                                mask[index]=fact_id

                    full_text_tensor = torch.tensor(ids_text)
                    self.raw_examples.append((full_text_tensor, mask, genre))
                    good_docs +=1
                except:
                    bad_docs +=1

            base_path = os.path.dirname(dir_path) if dir_path[-1] != "/" else os.path.dirname(dir_path[:-1])
            genre_list_path = os.path.join(base_path,"genre_list.npy")
            if is_train:
                GENRES_LIST = [k for k,v in genere_counter.items() if v > GENRE_APPEARENCE_MIN]
                np.save(genre_list_path,np.array(GENRES_LIST))
            else:
                GENRES_LIST = np.load(genre_list_path).tolist()


            for example in self.raw_examples:
                genre = example[2]
                if genre in GENRES_LIST:
                    genre_key = GENRES_LIST.index(genre)
                else:
                    genre_key = GENRES_LIST.index("unknown")

                self.examples.append((example[0], example[1], torch.tensor(genre_key)))

            logger.info("finished creating examples for " + dir_path)
            logger.info("docs turnes to examples = " + str(good_docs))
            logger.info("len of examples = " + str(len(self.examples)))
            logger.info("docs with facts = " + str(docs_with_facts_counter))
            logger.info("docs without enough facts = " + str(docs_without_facts_counter))
            logger.info("docs too long = " + str(long_docs))
            logger.info("docs with exceptions = " + str(bad_docs))
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
    mds = MaskedPlotFactEmbeddingDataset(tokenizer, args, path)
