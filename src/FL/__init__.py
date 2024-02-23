"""from src.SemaClassifier.classifier.GNN import GNN_script
from src.SemaClassifier.classifier.GNN import GINJKFlagClassifier
from src.SemaClassifier.classifier.GNN import utils

__all__ = [
    "GNN_script",
    "GINJKFlagClassifier",
    "utils"
]"""
import sys
import os
cwd=os.getcwd()
print(cwd,cwd+"/src")
sys.path.insert(0, cwd)