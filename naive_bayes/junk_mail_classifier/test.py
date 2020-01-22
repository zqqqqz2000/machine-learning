import time

from naive_bayes.junk_mail_classifier.classifier import *
from naive_bayes.junk_mail_classifier.prepare import *


if __name__ == '__main__':
    test_dataset, test_label = load_dataset('./data/test')
    train_dataset, train_label = load_dataset('./data/train')
    test_dataset = list(map(cut_and_remove, test_dataset))
    train_dataset = list(map(cut_and_remove, train_dataset))
    bayes = Classifier()
    bayes.fit(train_dataset, train_label)
    predict = bayes.predict(test_dataset)
    for data, label, label_true in zip(test_dataset, predict, test_label):
        print(f'predict: {label} label: {label_true} data: {data[:10]}')
    print(f'acc: {bayes.score(test_dataset, test_label)}')
