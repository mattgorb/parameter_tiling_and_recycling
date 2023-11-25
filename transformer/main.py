    
import argparse
import csv
import logging
import os
import random
import sys
import json

import numpy as np
import torch
from collections import namedtuple
from tempfile import TemporaryDirectory
from pathlib import Path
from torch.utils.data import (DataLoader, RandomSampler,Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import MSELoss

from file_utils import WEIGHTS_NAME, CONFIG_NAME
from transformer.transformer.bert_models import TinyBertForPreTraining, BertModel
from transformer.transformer.tokenization import BertTokenizer
from transformer.transformer.optimization import BertAdam


student_model=''#config.json
pretrained=True

if pretrained:
    #print("HEre?")
    student_model = TinyBertForPreTraining.from_pretrained(student_model)
else:
    student_model = TinyBertForPreTraining.from_scratch(student_model)

print(student_model)