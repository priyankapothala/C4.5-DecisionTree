import math
import numpy as np


# DecisionTreeNode class
class DecisionTreeNode:
    def __init__(self, attr, attr_type, decision=None, leaf_node=False, gain_ratio=None):
        self.attribute = attr
        self.attr_type = attr_type
        self.decision = decision
        self.isLeaf = leaf_node
        self.gain_ratio = gain_ratio
        self.children = []
        self.condition = []

    def to_string(self, depth=0):
        print_result = "\t"*depth+"{attribute:"+str(self.attribute) +\
            ",gain_ratio:"+str(self.gain_ratio) +\
            ",conditions:"+','.join(self.condition) +\
            ",attr_type:"+str(self.attr_type) +\
            ",decision:"+str(self.decision) +\
            ",isLeaf:"+str(self.isLeaf)+"}\n"
        for childNode in self.children:
            print_result += childNode.to_string(depth+1)
        return print_result

    def __str__(self):
        return self.to_string()

# Function for calculating total entropy


def total_entropy(data, label):
    entropy = 0
    total = data.shape[0]
    for index, value in data[label].value_counts().items():
        probability = (value/total)
        if probability != 0:
            entropy = entropy - (probability*math.log(probability, 2))
    return entropy

# Function for calculating gain ratio


def gain_ratio(data, attribute, class_label):
    info_data = total_entropy(data, class_label)
    attr_type = data[attribute].dtype
    if str(attr_type) == "category":
        split_info = 0
        info_attribute = 0
        total_count = data[attribute].shape[0]
        attr_values = data[attribute].unique()
        for value in attr_values:
            attr_value_total = data[data[attribute] == value].shape[0]
            partition = (attr_value_total/total_count)
            if partition != 0:
                split_info = split_info - (partition*math.log(partition, 2))
            entropy = 0
            for label in list(data[class_label].unique()):
                attr_class_total = data[(data[attribute] == value) & (
                    data[class_label] == label)].shape[0]
                probability = (attr_class_total/attr_value_total)
                if probability != 0:
                    entropy = entropy - (probability*math.log(probability, 2))
            info_attribute = info_attribute + \
                ((attr_value_total/total_count)*entropy)
        info_gain = info_data - info_attribute
        if split_info == 0:
            split_info = np.finfo(float).eps
        gain_ratio = info_gain/split_info
        return {'gain_ratio': gain_ratio}
    else:
        total_count = data[attribute].shape[0]
        attr_values = list(data[attribute].sort_values().unique())
        max_gain_ratio = float("-inf")
        split_value = None
        for i in range(len(attr_values)-1):
            cont_value = (attr_values[i]+attr_values[i+1])/2
            lt_count = data[data[attribute] <= cont_value].shape[0]
            gt_count = data[data[attribute] > cont_value].shape[0]
            lt_entropy = 0
            gt_entropy = 0
            for label in list(data[class_label].unique()):
                lt_atrr_count = data[(data[attribute] <= cont_value) & (
                    data[class_label] == label)].shape[0]
                gt_atrr_count = data[(data[attribute] > cont_value) & (
                    data[class_label] == label)].shape[0]
                lt_prob = (lt_atrr_count/lt_count)
                gt_prob = (gt_atrr_count/gt_count)
                if lt_prob != 0:
                    lt_entropy = lt_entropy - (lt_prob*math.log(lt_prob, 2))
                if gt_prob != 0:
                    gt_entropy = gt_entropy - (gt_prob*math.log(gt_prob, 2))

            info_gain = info_data - \
                ((lt_count/total_count)*lt_entropy) - \
                ((gt_count/total_count)*gt_entropy)
            lt_split, gt_split = (lt_count/total_count), (gt_count/total_count)
            split_info = 0
            if lt_split != 0:
                split_info = split_info - lt_split*math.log(lt_split, 2)
            if gt_split != 0:
                split_info = split_info - gt_split*math.log(gt_split, 2)
            if split_info == 0:
                split_info = np.finfo(float).eps
            gain_ratio = info_gain/split_info
            if gain_ratio > max_gain_ratio:
                max_gain_ratio = gain_ratio
                split_value = cont_value
        return {'gain_ratio': max_gain_ratio, 'split_value': split_value}


def buildTree(data, columns, attr_values, class_label, majority_label):
    # If the subset/partition is empty attach a leaf labeled with the majority class label
    if len(data) == 0:
        return DecisionTreeNode(attr=None, attr_type=None, leaf_node=True, decision=majority_label)

    # if all the records belong to the same class, return the majority class label
    elif len(list(data[class_label].unique())) == 1:
        majority_vote = data[class_label].value_counts().idxmax()
        return DecisionTreeNode(attr=None, attr_type=None, leaf_node=True, decision=majority_vote)

    # if attribute list is empty return the majority class label
    elif len(columns) == 0:
        return DecisionTreeNode(attr=None, attr_type=None, leaf_node=True, decision=majority_label)

    else:
        max_gain_ratio = float("-inf")
        attr_gain_ratio = None
        split_attribute = None
        split_attr_gain = None
        # Finding the attribute with maximum gain ratio
        for col in columns:
            attr_gain_ratio = gain_ratio(data, col, class_label)
            curr_gain_ratio = attr_gain_ratio['gain_ratio']
            if curr_gain_ratio > max_gain_ratio:
                max_gain_ratio = curr_gain_ratio
                split_attribute = col
                split_attr_gain = attr_gain_ratio

        # Removing the attribute with the maximum gain ratio
        remainingColumns = columns[:]
        remainingColumns.remove(str(split_attribute))

        attr_type = 'category' if str(
            data[split_attribute].dtype) == 'category' else 'continuous'
        DSNode = DecisionTreeNode(
            attr=split_attribute, attr_type=attr_type, gain_ratio=max_gain_ratio)

        if str(data[split_attribute].dtype) != 'category':
            condition = []
            condition.append(' <= '+str(split_attr_gain['split_value']))
            condition.append(' > '+str(split_attr_gain['split_value']))
            DSNode.condition = condition
            subsets = []
            subsets.append(data[data[split_attribute] <=
                                split_attr_gain['split_value']])
            subsets.append(data[data[split_attribute] >
                                split_attr_gain['split_value']])
            for subset in subsets:
                DSNode.children.append(
                    buildTree(subset, remainingColumns, attr_values, class_label, majority_label))
        else:
            condition = []
            subsets = []
            for attr_value in attr_values[split_attribute]:
                condition.append(' == "'+str(attr_value)+'"')
                subsets.append(data[data[split_attribute] == attr_value])
            DSNode.condition = condition
            for subset in subsets:
                DSNode.children.append(
                    buildTree(subset, remainingColumns, attr_values, class_label, majority_label))

        return DSNode

# Function for prediction


def predict_label(row, node):
    if node.isLeaf:
        return node.decision
    else:
        for index, condition in enumerate(node.condition):
            if node.attr_type == "category":
                condition_string = '"'+str(row[node.attribute])+'"' + condition
            else:
                condition_string = str(row[node.attribute]) + condition
            if eval(condition_string):
                return predict_label(row, node.children[index])
