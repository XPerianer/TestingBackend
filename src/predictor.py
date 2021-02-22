#
# The MIT License (MIT)
#
# Copyright (c) 2020-2021 Dominik Meier
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
from abc import ABC, abstractmethod


class Predictor(ABC):
    """Abstract Base Class for Predictors"""

    @abstractmethod
    def name(self):
        """Output a concise name that makes it easy to identify"""
        raise NotImplementedError

    @abstractmethod
    def fit(self, X_train, y_train):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X_test):
        raise NotImplementedError


class NearestMutantPredictor(Predictor):
    """Chooses the same test outcomes as the mutant with the smallest absolute difference in mutant_id"""

    def name(self):
        return "NearestMutant"

    def fit(self, X_train, y_train):
        self.X = X_train.copy()
        self.X["outcome"] = y_train

    def predict(self, X_test):
        predictions = []
        for index, row in X_test.iterrows():
            # Select only rows from X_train with the same test_id
            correct_tests = self.X.loc[self.X["test_id"] == row["test_id"]]
            mutant_id = row["mutant_id"]
            nearest_mutant_id_index = abs(
                correct_tests["mutant_id"] - mutant_id
            ).idxmin()
            predictions.append(self.X["outcome"][nearest_mutant_id_index])
        return predictions
