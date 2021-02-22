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
import numpy as np


# TODO: This could really good live in a class
# This prepares a dataset that can be used to quickly look up the tf-idf score of
# a specific combination of test and filepath
# It returns a series with test_names and modified_file_path as a hierarchical index to the tfidf
def tf_idf_preparation(data):
    test_failures = data.loc[data["outcome"] is False]
    N = len(data.groupby("filepath"))
    idf_counts = (
        test_failures.groupby(["full_name", "modified_file_path"])
        .count()
        .groupby(["full_name"])
        .count()["mutant_id"]
    )
    idf = np.log(1 + N / idf_counts)
    tf = np.log(1 + test_failures.groupby(["full_name", "modified_file_path"]).count())
    join = tf.join(idf, lsuffix="_tf")
    tfidf = join["mutant_id_tf"] * join["mutant_id"]
    print(tfidf)
    return tfidf


def tf_idf_from_file_path(file_path, tfidf_data):
    return tfidf_data.xs(file_path, level=1)
