import numpy as np

sys.path.append("../")
from utils.data_utils import get_data_for

_, y = get_data_for("tree", "train")

random = np.random.choice([0, 1], size=len(y))
random_prob = np.random.rand(len(y))

print("bal acc: ", balanced_accuracy_score(y, random))
print("auc roc: ", roc_auc_score(y, random_prob))
