from transformers import XLNetConfig


class CPXLNetConfig(XLNetConfig):
    def __init__(self, n_genres=None, n_facts=None, **kwargs):
        super().__init__(**kwargs)
        self.n_genres = n_genres
        self.n_facts = n_facts