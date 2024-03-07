#!/usr/bin/python
# encoding: utf-8

import collections

import numpy as np

class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1


    #it encodes the text into a list of integers. It simulates the CTC loss function
    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            np.array [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            np.array [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (np.array(text, dtype=np.int32), np.array(length, dtype=np.int32))


    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            np.ndarray [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            np.ndarray [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length do not match.
        Returns:
            text (str or list of str): texts to convert.
        """

        if len(length) == 1:
            length = length[0]
            assert len(t) == length, "text with length: {} does not match declared length: {}".format(len(t), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert len(t) == length.sum(), "texts with length: {} does not match declared length: {}".format(len(t), length.sum())
            texts = []
            index = 0
            for i in range(len(length)):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], np.array([l]), raw=raw))
                index += l
            return texts


a = strLabelConverter("0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~!\"#$%&'()*+,-./ ")
print(a.encode("Helo"))
print(a.decode(a.encode("Welcome to the world of tomorrow!")[0], raw=False, length=a.encode("Welcome to the world of tomorrow!")[1]))