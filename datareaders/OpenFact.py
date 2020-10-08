import json
from typing import List, ClassVar, Type

from marshmallow import Schema
from marshmallow_dataclass import dataclass

@dataclass
class Tok:
    word : str
    start : int
    end : int

@dataclass
class OpenFact:
        text : str
        position : int
        subject : str
        subjectBegin : int
        subjectEnd : int
        relation :str
        relationBegin : int
        relationEnd : int
        obj :str
        objBegin : int
        objEnd : int
        salience :float
        token: List[Tok]

@dataclass
class FactsDoc:
    docID : str
    text : str
    openfacts: List[OpenFact]
    Release_Year: int
    Title: str
    Origin_Ethnicity: str
    Director: str
    Cast: str
    Genre: str
    Wiki_Page: str
    Schema: ClassVar[Type[Schema]] = Schema

# class OpenFactTensors:
#         def __init__(self, subject, relation, obj, norm_salience, position):
#             self.norm_salience = norm_salience
#             self.obj = obj
#             self.relation = relation
#             self.subject = subject
#             self.postion = position
#
#
# class FactsDocTensors:
#         def __init__(self, text, openFacts):
#             self.openFacts = openFacts
#             self.text = text

#test
if __name__ == '__main__':
    path ="/Users/eyalorbach/Projects/thesis-tests/playground/data/plots_salie_modified/salie_plots_parsed/valid/valid_0.json"
    with open(path) as f:
        content = json.load(f)
        data = FactsDoc.Schema().load(content)
    a =data
    pass