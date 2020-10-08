
SRC_TOKENIZER_SPECIAL_CHAR = b'\xe2\x96\x81'
HELPER_TOKENIZER_SPECIAL_CHAR =b'\xc4\xa0'

class TokenizationMapper(object):
    mapping = {}
    target_unk_idx = None


    def get_target_idx_From_src(self, src_idx):
        return self.mapping.get(src_idx, self.target_unk_idx)


    def init_map(self, source_tokenizer, target_tokenizer):
        self.target_unk_idx = target_tokenizer.encoder.get(target_tokenizer.unk_token)
        for i in range(source_tokenizer.vocab_size):
            tok_arr = source_tokenizer.convert_ids_to_tokens([i])
            if len(tok_arr) is not 1:
                tok = target_tokenizer.unk_token
            else:
                tok = tok_arr[0]


            if tok.startswith(SRC_TOKENIZER_SPECIAL_CHAR.decode('utf-8')):
                newtok = HELPER_TOKENIZER_SPECIAL_CHAR.decode('utf-8') + tok[1:]
            else:
                newtok = tok

            target_idx_arr = target_tokenizer.convert_tokens_to_ids([newtok])
            if len(target_idx_arr) is not 1:
                self.mapping[i] = self.target_unk_idx
            else:
                self.mapping[i] = target_idx_arr[0]
