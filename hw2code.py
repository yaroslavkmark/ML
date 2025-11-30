import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
    if len(np.unique(feature_vector)) <= 1:
        return np.array([]), np.array([]), None, None

    sorted_indices = np.argsort(feature_vector)
    sorted_features = feature_vector[sorted_indices]
    sorted_targets = target_vector[sorted_indices]

    # Более надежный способ найти места изменения значений
    diff_indices = np.where(np.diff(sorted_features) != 0)[0]
    if len(diff_indices) == 0:
        return np.array([]), np.array([]), None, None

    thresholds = (sorted_features[diff_indices] + sorted_features[diff_indices + 1]) / 2

    n_total = len(target_vector)
    ones_cumsum = np.cumsum(sorted_targets)

    left_ones = ones_cumsum[diff_indices]
    left_sizes = diff_indices + 1
    left_zeros = left_sizes - left_ones

    total_ones = ones_cumsum[-1]
    right_ones = total_ones - left_ones
    right_sizes = n_total - left_sizes
    right_zeros = right_sizes - right_ones

    # Защита от деления на ноль
    eps = 1e-9
    h_left = 1 - (left_ones / (left_sizes + eps)) ** 2 - (left_zeros / (left_sizes + eps)) ** 2
    h_right = 1 - (right_ones / (right_sizes + eps)) ** 2 - (right_zeros / (right_sizes + eps)) ** 2

    ginis = -(left_sizes / n_total) * h_left - (right_sizes / n_total) * h_right

    # Выбор лучшего порога с учетом требования "минимальный сплит при равенстве"
    best_idx = np.argmax(ginis)

    return thresholds, ginis, thresholds[best_idx], ginis[best_idx]


class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if len(sub_y) == 0:
            node["type"] = "terminal"
            node["class"] = 0
            return

        # Все объекты одного класса
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        # Проверка максимальной глубины
        if self._max_depth is not None and depth >= self._max_depth:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # Проверка min_samples_split
        if self._min_samples_split is not None and len(sub_y) < self._min_samples_split:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        # Проверка min_samples_leaf для терминального узла
        if self._min_samples_leaf is not None and len(sub_y) < 2 * self._min_samples_leaf:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None

        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]

            if feature_type == "real":
                feature_vector = sub_X[:, feature]

            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}

                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0

                    ratio[key] = current_click / (current_count + 1e-9)

                sorted_categories = sorted(ratio.items(), key=lambda x: x[1])
                categories_map = {cat: idx for idx, (cat, _) in enumerate(sorted_categories)}

                feature_vector = np.array([categories_map[x] for x in sub_X[:, feature]])

            else:
                raise ValueError(f"Unknown feature type: {feature_type}")

            if len(np.unique(feature_vector)) <= 1:
                continue

            thresholds, ginis, threshold, gini = find_best_split(feature_vector, sub_y)

            if gini is None:
                continue

            # Проверяем ВСЕ возможные разбиения на соответствие min_samples_leaf
            for i, thresh in enumerate(thresholds):
                current_split = feature_vector < thresh
                left_samples = np.sum(current_split)
                right_samples = len(sub_y) - left_samples

                # Проверка min_samples_leaf
                if (self._min_samples_leaf is not None and
                        (left_samples < self._min_samples_leaf or right_samples < self._min_samples_leaf)):
                    continue  # Пропускаем это разбиение

                current_gini = ginis[i]

                if gini_best is None or current_gini > gini_best:
                    feature_best = feature
                    gini_best = current_gini
                    threshold_best = thresh
                    split = current_split

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best

        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            # Восстанавливаем список категорий для левого поддерева
            if self._feature_types[feature_best] == "categorical":
                counts = Counter(sub_X[:, feature_best])
                clicks = Counter(sub_X[sub_y == 1, feature_best])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / (current_count + 1e-9)
                sorted_categories = sorted(ratio.items(), key=lambda x: x[1])
                categories_map = {cat: idx for idx, (cat, _) in enumerate(sorted_categories)}
                threshold_best = [cat for cat, idx in categories_map.items() if idx < threshold_best]
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[~split], sub_y[~split], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        # ╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ
        if node["type"] == "terminal":
            return node["class"]

            # Нетерминальный узел - продолжаем спуск
        feature_idx = node["feature_split"]
        feature_type = self._feature_types[feature_idx]

        if feature_type == "real":
            # Вещественный признак
            if x[feature_idx] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        elif feature_type == "categorical":
            # Категориальный признак
            if x[feature_idx] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

        else:
            raise ValueError(f"Unknown feature type: {feature_type}")

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)
