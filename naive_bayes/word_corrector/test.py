from naive_bayes.word_corrector.corrector import Corrector

correction: callable


def unit_tests(words_filename: str):
    global correction
    correction = Corrector(words_filename).correction
    assert correction('speling') == 'spelling'  # insert
    assert correction('korrectud') == 'corrected'  # replace 2
    assert correction('bycycle') == 'bicycle'  # replace
    assert correction('inconvient') == 'inconvenient'  # insert 2
    assert correction('arrainged') == 'arranged'  # delete
    assert correction('peotry') == 'poetry'  # transpose
    assert correction('peotryy') == 'poetry'  # transpose + delete
    assert correction('word') == 'word'  # known
    assert correction('quintessential') == 'quintessential'  # unknown
    return 'unit_tests pass'


if __name__ == '__main__':
    print(unit_tests('data/words.txt'))
    while True:
        print(correction(input('输入要纠正的词: ')))
