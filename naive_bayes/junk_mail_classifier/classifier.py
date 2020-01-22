import math
from typing import *
from sklearn_style import SklearnStyle


class Classifier(SklearnStyle):
    """Text classifier use naive Bayes"""
    def __init__(self, laplace_smoothing: bool=True):
        """
        init func of classifier
        :param laplace_smoothing: if use laplace smoothing, default is True
        """
        self.lambda_: float = 1 if laplace_smoothing else 0
        self.P: Dict[str, float] = {}
        self.words_in_categories_num: Dict[str, int] = {}
        self.categories: Set[str] = set()

    def fit(self, texts: Sequence[Sequence[str]], categories: Sequence[str]) -> NoReturn:
        """
        input the texts and the categories of texts to learn the model
        :param texts: the type of this param is Sequence[Sequence[str]]
        :param categories: the type of this param is Sequence[str]
        """
        for category in categories:
            self.P[category] = self.P.get(category, 0) + 1
            self.categories.add(category)
        for mail, category in zip(texts, categories):
            for word in mail:
                self.P[f'{word}|{category}'] = self.P.get(f'{word}|{category}', 0) + 1
                self.words_in_categories_num[category] = self.words_in_categories_num.get(category, 0) + 1
        for key in self.P.keys():
            if '|' in key:
                self.P[key] = (self.P[key] + self.lambda_) / (self.words_in_categories_num[key.split('|')[1]] + 2 * self.lambda_)
            else:
                self.P[key] = (self.P[key] + self.lambda_) / (2 * self.lambda_ + len(categories))
            self.P[key] = math.log2(self.P[key])

    def score(self, x: any, y: any) -> Sequence[float]:
        """
        call predict and input x to calculate the y^(predict categories) and compare to the y to calculate the acc
        :param x: input data
        :param y: the true category
        :return: the sequence of some accurate score
        """
        right = 0
        res: List[str] = self.predict(x)
        for y__, y_ in zip(res, y):
            if y__ == y_:
                right += 1
        return [right / len(res)]

    def predict(self, x: Sequence[Sequence[str]]) -> List[str]:
        """
        input data and using fit model to predict y
        :param x: x is some text, which type is Sequence[Sequence[str]]
        :return: return the categories of each text in sequence
        """
        res: List[str] = []
        for simple in x:
            category_probability: Dict[str: float] = {}
            for category in self.categories:
                category_probability[category] = self.P[category]
                for word in simple:
                    category_probability[category] += self.P.get(f'{word}|{category}', math.log2(0.5))
            res.append(max(category_probability.keys(), key=lambda key: category_probability[key]))
        return res
