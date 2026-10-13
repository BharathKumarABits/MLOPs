"""Fail the pipeline if the freshly trained model is below the quality bar."""
import json
import sys

MIN_TEST_ROC_AUC = 0.80

card = json.load(open("models/model_card.json"))
auc = card["test"]["roc_auc"]
print(f"selected={card['selected_model']} test_roc_auc={auc}")
if auc < MIN_TEST_ROC_AUC:
    sys.exit(f"Quality gate failed: ROC-AUC {auc} < {MIN_TEST_ROC_AUC}")
