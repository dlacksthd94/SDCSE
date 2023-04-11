from dataclasses import dataclass, field
from transformers import HfArgumentParser, TrainingArguments
import sys

@dataclass
class TestArguments:
    i: int = field(
        default=1,
        init=True,
        metadata={
            'help': 'arg i'
        }
    )
    
    s: str = field(
        default='x',
        init=True,
        metadata={
            'help': 'arg s'
        }
    )

print(sys.argv)
sys.argv = ['train.py', '--i', '2']

dataclass_types = [TestArguments]
dtype = dataclass_types[0]
parser = HfArgumentParser((TestArguments, ))
self=parser
parser.parse_args_into_dataclasses()

