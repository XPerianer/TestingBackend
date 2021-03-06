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
from typing import List

import numpy as np
from pandas import DataFrame


class ReorderingEvaluation:
    def __init__(self, ordering: List[int], dataframe: DataFrame):
        self.ordering = ordering
        self.number_of_tests = len(self.ordering)
        self.dataframe = dataframe
        self.number_of_failed_tests = self.dataframe.loc[
            self.dataframe["outcome"] is False
        ].shape[0]
        # TODO: assert that ordering and dataframe are kind of legit

    def to_dict(self):
        return {
            "APFD": self.APFD(),
            "APFDc": self.APFDc(),
            "first_failing_duration": self.first_failing_duration(),
            "last_failing_duration": self.last_test_failing_duration(),
        }

    def first_failing_duration(self):
        summed_duration = 0
        for test_id in self.ordering:
            rows = self.dataframe.loc[self.dataframe["test_id"] == test_id]
            assert (
                rows.shape[0] == 1
            ), "The provided dataframe contains more than one entry for the given test_id"

            summed_duration += rows["duration"].iloc[0]
            if rows["outcome"].iloc[0] is False:
                return summed_duration
        return summed_duration

    def last_test_failing_duration(self):
        summed_duration = 0
        temporary_duration = 0
        for test_id in self.ordering:
            rows = self.dataframe.loc[self.dataframe["test_id"] == test_id]
            assert (
                rows.shape[0] == 1
            ), "The provided dataframe contains more than one entry for the given test_id"

            if rows["outcome"].iloc[0] is True:
                temporary_duration += rows["duration"].iloc[0]
            if rows["outcome"].iloc[0] is False:
                summed_duration += temporary_duration + rows["duration"].iloc[0]
                temporary_duration = 0

        return summed_duration

    def APFD(self):
        """Calculates the widespread used Average Percentage of Faults detected metric
        We have the simplification here, that we say that each test failure is a defect,
        anc vice versa, so the number of failed tests is the number of faults
        While this is not realisitic, as probably some integration tests test the same as others,
        it is easier to calculate
        """

        number_encoded_test_outcomes = [
            0
        ]  # We always start at the beginning with 0 known failures and 0 executed tests
        for test_id in self.ordering:
            number_encoded_test_outcomes.append(
                not self.dataframe.loc[
                    self.dataframe["test_id"] == test_id, "outcome"
                ].values[0]
            )

        summed_number_encoded_test_outcome = (
            np.cumsum(number_encoded_test_outcomes) * 1 / self.number_of_failed_tests
        )

        return np.trapz(
            summed_number_encoded_test_outcome,
            np.linspace(0, 1, num=(self.number_of_tests + 1)),
        )

    def APFDc(self):
        """Similar to APFD, but takes the average duration of the tests into account"""

        number_encoded_test_outcomes = [
            0
        ]  # We always start at the beginning with 0 known failures and 0 executed tests
        durations_test_outcomes = [0]
        for test_id in self.ordering:
            number_encoded_test_outcomes.append(
                not self.dataframe.loc[
                    self.dataframe["test_id"] == test_id, "outcome"
                ].values[0]
            )
            durations_test_outcomes.append(
                self.dataframe.loc[
                    self.dataframe["test_id"] == test_id, "duration"
                ].values[0]
            )

        summed_number_encoded_test_outcome = (
            np.cumsum(number_encoded_test_outcomes) * 1 / self.number_of_failed_tests
        )
        summed_durations_test_outcomes = (
            np.cumsum(durations_test_outcomes) * 1 / np.sum(durations_test_outcomes)
        )

        return np.trapz(
            summed_number_encoded_test_outcome, summed_durations_test_outcomes
        )
