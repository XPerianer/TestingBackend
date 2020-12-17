from git import Repo
from unidiff import PatchSet
from pandas import DataFrame

#encoded_column_names = ["modified_method", "modified_file_path", "name", "filepath", "current_line", "previous_line"]
encoded_column_names = ["modified_file_path"]#, "previous_line"]
dangerous_features = ['duration', 'setup_outcome', 'setup_duration', 'call_outcome', 'call_duration', 'teardown_outcome', 'teardown_duration']
unencoded_features = ['repo_path', 'full_name']

def predict_failing_tests(path, predictor, encoder, test_ids_to_test_names):
    # Get Change from filename
    repo = Repo(path)
    diff = repo.git.diff(repo.head, None, '--unified=0')
    patchset = PatchSet(diff)
    # Analyze all the different changes
    mutants = []
    for patchedFile in patchset:
        for hunk in patchedFile:
            line_difference = 0
            for line in hunk:
                line_difference += 1
                if line.is_removed:
                    mutants.append({
                        'modified_file_path': patchedFile.target_file[2:],  # the [2:] removes unwanted prefixes
                        #'previous_line': str(line)[2:],
                        'line_number_changed' : hunk.source_start + line_difference,
                    })

    test_ids = test_ids_to_test_names.index
    mutants_with_test_ids = []
    for mutant in mutants:
        for test_id in test_ids:
            mutant['test_id'] = test_id
            mutants_with_test_ids.append(mutant.copy())                
    
    mutants_with_test_ids = DataFrame(mutants_with_test_ids)
    mutants_with_test_ids
    mutants_with_test_ids[encoded_column_names] = encoder.transform(mutants_with_test_ids[encoded_column_names])
    mutants_with_test_ids['prediction'] = predictor.predict(mutants_with_test_ids)
    # Union all predicted failures
    prediction_per_test_id = mutants_with_test_ids.groupby('test_id').all()['prediction']

    return prediction_per_test_id[prediction_per_test_id == False].index
