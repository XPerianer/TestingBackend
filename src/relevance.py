import numpy as np

# TODO: This could really good live in a class
# This prepares a dataset that can be used to quickly look up the tf-idf score of a specific combination of test and filepath
# It returns a series with test_names and modified_file_path as a hierarchical index to the tfidf
def tf_idf_preparation(data):
    test_failures = data.loc[data['outcome'] == False]
    N = len(data.groupby('filepath'))
    idf_counts = test_failures.groupby(['full_name', 'modified_file_path']).count().groupby(['full_name']).count()['mutant_id']
    idf = np.log(1 + N / idf_counts)
    tf = np.log( 1 + test_failures.groupby(['full_name', 'modified_file_path']).count())
    join = tf.join(idf, lsuffix='_tf')
    tfidf = join['mutant_id_tf'] * join['mutant_id']
    print(tfidf)
    return tfidf

def tf_idf_from_file_path(file_path, tfidf_data):
    return tfidf_data.xs(file_path, level=1)
