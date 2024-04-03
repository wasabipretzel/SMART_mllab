import os
import shutil
import json
import numpy as np



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
