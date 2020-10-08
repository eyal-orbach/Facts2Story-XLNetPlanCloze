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
from sentence_splitter import SentenceSplitter, split_text_into_sentences
FACTS_NUM = 5
SENTENCE_NUM = 5


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


class MaskedPlotShortenedDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, dir_path: str, block_size=1024):
        self.examples=[]
        tokenizer_class = tokenizer.__class__.__name__
        cached_features_file = os.path.join(
            dir_path, args.model_type + "_cache43shrt_maskedplm2_" + str(block_size) + "_" + tokenizer_class
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", dir_path)

            docs_with_facts_counter = docs_without_facts_counter = long_docs = bad_docs = 0
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

                    if encoded_text.size(-1) > (block_size-20):
                        long_docs+=1
                        continue

                    selected_facts = sorted(facts_doc.openfacts, key=lambda x: x.relationEnd)[:1]

                    prefix_tokens = []#[tokenizer.additional_special_tokens_ids[1]]
                    ids_text_no_prefix = tokenizer.convert_tokens_to_ids(tokenized_text)
                    ids_text = prefix_tokens + ids_text_no_prefix

                    strpos2index = get_strpos2index(tokenizer, encoded_text, len(prefix_tokens))
                    splitter = SentenceSplitter(language='en')
                    full_text_sentence_split = splitter.split(text=facts_doc.text)

                    partial_text_sent_split = full_text_sentence_split[:SENTENCE_NUM+1]

                    partial_encoded_sent_split= [tokenizer.encode(t, add_special_tokens=False, return_tensors="pt") \
                        .squeeze(0) for t in partial_text_sent_split]

                    partial_encoded = torch.cat(partial_encoded_sent_split, dim=0)
                    mask = torch.zeros(partial_encoded.size())
                    for i in range(len(prefix_tokens)):
                        mask[i]=1

                    for fact in selected_facts:
                        for tok in fact.token:
                            for i in range(tok.start, tok.end):
                                if i not in strpos2index:
                                    print (f"{i} out of index in {tok} from {fact.text}")
                                    raise LookupError
                                index = strpos2index[i]
                                mask[index]=1
                    masked_sent=0
                    for i,sent in enumerate(partial_encoded_sent_split):
                        start = sum([s.size(0) for s in partial_encoded_sent_split[:i]])
                        end = start + sent.size(0)
                        if mask[start:end].sum() != 0:
                            masked_sent = i

                    if masked_sent == 0:
                        sents_to_expose = [1,5]
                    elif masked_sent== 5:
                        sents_to_expose = [0,4]
                    else:
                        sents_to_expose = [0,5]

                    for sent_idx in sents_to_expose:
                        start = sum([s.size(0) for s in partial_encoded_sent_split[:sent_idx]])
                        end = start + partial_encoded_sent_split[sent_idx].size(0)
                        mask[start:end] =1

                    full_text_tensor = torch.tensor(partial_encoded)
                    self.examples.append((full_text_tensor, mask))
                except:
                    bad_docs +=1


            logger.info("finished creating examples for " + dir_path)
            logger.info("docs with facts = " + str(docs_with_facts_counter))
            logger.info("docs without enough facts = " + str(docs_without_facts_counter))
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

    path = "/Users/eyalorbach/Projects/thesis-tests/playground/data/json_out_details_server"
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', cache_dir="/tmp/cache2")
    mds = MaskedPlotShortenedDataset(tokenizer, args, path)
