import pytest

from app import app


@pytest.fixture
def client():

    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# Test if the testfile json is received as an array of tests that have f.e. the mutant_failures option
def test_json_serving(client):
    return_value = client.get('/data')
    assert len(return_value.get_json()) == 455
    assert return_value.get_json()[0]['mutant_failures'] == 77
