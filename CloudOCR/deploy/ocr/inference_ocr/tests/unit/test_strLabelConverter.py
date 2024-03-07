import pytest

from inference_ocr.adapters.models.utils import strLabelConverter
import numpy as np

@pytest.fixture
def converter():
    alphabet = """0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~!"#$%&'()*+,-./ """
    converter = strLabelConverter(alphabet)
    return converter

def test_strLabelConverter(converter):
    assert converter.decode(converter.encode("Hell-loo World")[0], raw=False, length=converter.encode("Hell-loo World")[1]) == "hel-lo world"
    assert converter.decode(converter.encode("Hell-loo World")[0], raw=True, length=converter.encode("Hell-loo World")[1]) == "hell-loo world"
    assert converter.decode(converter.encode("Welcome to the world of tomorrow!")[0], raw=False, length=converter.encode("Welcome to the world of tomorrow!")[1]) == "welcome to the world of tomorow!"
    assert converter.decode(converter.encode("Welcome to the worrrrrld of tomorrow!")[0], raw=True, length=converter.encode("Welcome to the worrrrrld of tomorrow!")[1]) == "welcome to the worrrrrld of tomorrow!"
    assert converter.decode(converter.encode("Welcome to the worrrrrld of tomorrow!")[0], raw=False, length=converter.encode("Welcome to the worrrrrld of tomorrow!")[1]) == "welcome to the world of tomorow!"
    assert converter.decode(converter.encode("Welcome to the worrrrrld of tomoooooor-row!")[0], raw=True, length=converter.encode("Welcome to the worrrrrld of tomoooooor-row!")[1]) == "welcome to the worrrrrld of tomoooooor-row!"
    