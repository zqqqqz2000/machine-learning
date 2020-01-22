import abc
from abc import ABCMeta
from typing import *
from numpy import ndarray


class SklearnStyle(metaclass=ABCMeta):
    @abc.abstractmethod
    def fit(self, *args, **kwargs) -> NoReturn:
        """
        this func is used to train the model
        :param args: some data use to train the model
        :param kwargs: some options to control the training of the model
        :return void func, NoReturn
        """
        pass

    @abc.abstractmethod
    def score(self, x: any, y: any) -> Union[Iterable[float], ndarray]:
        """
        score method must be defined to check the accuracy
        :param x: data to calculate the y^
        :param y: data to verify the accuracy of the model
        :return Iterable object which contains float object, or return a ndarray type of numpy
        """
        pass

    @abc.abstractmethod
    def predict(self, x: any) -> any:
        """
        input x to calculate the y^
        :param x: data source to predict the y
        :return return the y^, which can be any type
        """
        pass
