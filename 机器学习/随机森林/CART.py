def cal_gini_index(data):
    total_sample = len(data)
    if len(data) == 0:
        return 0
    label_counts = label_uniq_cnt(data)
    gini = 0
    for label in label_counts:
        gini = gini + pow(label_counts[label], 2)
    gini = 1 - float(gini) / pow(total_sample, 2)
    return gini

def label_uniq_cnt(data):
    label_uniq_cnt = {}
    for x in data:
        label = x[len(x) - 1]
        if label not in label_uniq_cnt:
            label_uniq_cnt[label] = 0
        label_uniq_cnt[label] = label_uniq_cnt[label] + 1
    return label_uniq_cnt

class node:
    def __init__(self, fea=-1, value=None, results=None, right=None, left=None):
        self.fea = fea
        self.value = value
        self.results = results
        self.right = right
        self.left = left

def split_tree(data, fea, value):
    set_1 = []
    set_2 = []
    for x in data:
        if x[fea] >= value:
            set_1.append(x)
        else:
            set_2.append(x)
    return (set_1, set_2)
        
def build_tree(data):
    if len(data) == 0:
        return node()
    currentGini = cal_gini_index(data)
    bestGain = 0.0
    bestCriteria = None
    bestSets = None
    feature_num = len(data[0]) - 1
    for fea in range(0, feature_num):
        feature_values = {}
        for sample in data:
            feature_values[sample[fea]] = 1
        for value in feature_values.keys():
            (set_1, set_2) = split_tree(data, fea, value)
            nowGini = float(len(set_1) * cal_gini_index(set_1) +
                            len(set_2) * cal_gini_index(set_2)) / len(data)
            gain = currentGini - nowGini
            if gain > bestGain and len(set_1) > 0 and len(set_2) > 0:
                bestGain = gain
                bestCriteria = (fea, value)
                bestSets = (set_1, set_2)
    if bestGain > 0:
        right = build_tree(bestSets[0])
        left = build_tree(bestSets[1])
        return node(fea=bestCriteria[0], value=bestCriteria[1], right=right, left=left)
    else:
        return node(results=label_uniq_cnt(data))
    
def predict(sample, tree):
    if tree.results != None:
        return tree.results
    else:
        val_sample = sample[tree.fea]
        branch = None
        if val_sample >= tree.value:
            branch = tree.right
        else:
            branch = tree.left
        return predict(sample, branch)
