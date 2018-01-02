import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_sample_image
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import matplotlib.pyplot as plt


def gini_index(labels):
    """
    计算基尼系数gini_index = 1- sum(pi**2)
    :param labels:
    :return:
    """

    _, ue_counts = np.unique(labels,return_counts=True)
    p = ue_counts/len(labels)
    gini = 1-sum(p**2)
    return gini


def choose_best_feature_index(observations, labels):
    """
    feature_index 分支
    :param observations:
    :param labels:
    :return:
    """

    nfeature = observations.shape[1]
    best_fval = np.zeros(nfeature)
    gini = np.zeros(nfeature)
    for feature_index in range(nfeature):
        feature = observations[:, feature_index]
        best_fval[feature_index],  gini[feature_index] = choose_best_feature_value(feature, labels)
    best_feature_index = np.argmin(gini)
    min_gini = np.min(gini)
    best_feature_value = best_fval[best_feature_index]
    return best_feature_value, best_feature_index, min_gini


def choose_best_feature_value(feature, labels):
    """
    根据某特征值进行分支
    :param feature:
    :param labels:
    :return:
    """
    uf = np.unique(feature)
    total_gini = np.zeros_like(uf)
    for i in range(len(uf)):
        f_val = uf[i]
        mask = feature < f_val
        left = labels[mask]
        right = labels[~mask]
        weights = np.array([len(left), len(right)])/len(labels)
        total_gini[i] = weights.dot([gini_index(left), gini_index(right)])
    best_fval = uf[np.argmin(total_gini)]
    min_gini = np.min(total_gini)
    return best_fval, min_gini


def is_leaf(node, cur_depth, max_depth=10, min_leaf_size=3):
    """
        判断一个节点是不是叶子节点
        cond1: cru_depth>=max_depth
        cond2: right or left is empty
        cond3: any([len(right[1]),len(left[1])]<min_leaf_size)
        条件2和条件3重复了
    """
    left = node['left']
    right = node['right']
    cond1 = cur_depth > max_depth
    cond2 = any(np.array([len(right[1]), len(left[1])]) < min_leaf_size)
    return cond1 or cond2


def split_data_into_node(observations, labels):
    """
    make data into a node: split the data
    :param observations:
    :param labels:
    :return:
    """
    best_fval, best_fi, gini = choose_best_feature_index(observations, labels)
    right_mask = observations[:, best_fi] >= best_fval
    right = [observations[right_mask, :], labels[right_mask]]
    left = [observations[~right_mask, :], labels[~right_mask]]
    node = {'left': left, 'right': right, 'gini': gini, 'findex': best_fi, 'fval': best_fval}
    return node


def leaf_prediction(leaf_node):
    """
    make prediction on leaf_node
    :param leaf_node:
    :return:
    """
    labels = list(np.hstack([leaf_node['left'][1], leaf_node['right'][1]]))
    pred = max(labels, key=labels.count)
    leaf_node['pred'] = pred
    return pred


def grow_tree(node, max_depth=50, min_leaf_size=1, cur_depth=1):
    """
    递归建树, 从一个节点开始长树
    :param node:
    :param max_depth:
    :param min_leaf_size:
    :param cur_depth:
    :return:
    """

    sp = split_data_into_node
    # 判断是否是叶子节点
    if is_leaf(node, cur_depth, max_depth=max_depth, min_leaf_size=min_leaf_size):
        pred = leaf_prediction(node)
        node['pred'] = pred

    else:
         left_node, right_node = sp(*node['left']), sp(*node['right'])
         cur_depth = cur_depth+1
         node['left'] = grow_tree(left_node, max_depth=max_depth, min_leaf_size=min_leaf_size, cur_depth=cur_depth)
         node['right'] = grow_tree(right_node, max_depth=max_depth, min_leaf_size=min_leaf_size, cur_depth=cur_depth)

    return node


def make_prediction(observation, tree):
    """
    make prediction on new observations
    :param observation:
    :param tree:
    :return:
    """

    fval = observation[tree['findex']]
    if fval < tree['fval']:
        if 'pred' in tree['left'].keys():
            pred = tree['left']['pred']
        else:
            pred = make_prediction(observation, tree['left'])
    else:
        if 'pred' in tree['right'].keys():
            pred = tree['right']['pred']
        else:
            pred = make_prediction(observation, tree['right'])
    return pred


def main():
    from sklearn.datasets import load_iris
    iris = load_iris()
    observations = iris.data
    labels = iris.target
    root = split_data_into_node(observations, labels)
    root = grow_tree(root)
    preds = []
    for i in range(len(labels)):
        preds.append(make_prediction(observations[i], root))


if __name__ == '__main__':
    main()
