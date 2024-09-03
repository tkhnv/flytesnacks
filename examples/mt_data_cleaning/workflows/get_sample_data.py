from flytekit import task


@task()
def get_sample_data() -> list[dict]:
    """Get sample data for the mt_data_cleaning example."""
    return [
        {"source": "Hello, world!", "target": "Bonjour, le monde!"},
        {"source": "This is a test.", "target": "C'est un test."},
        {"source": "Goodbye!", "target": "Au revoir!"},
        {"source": "Goodbye!", "target": "Very very very very very very long sentence."},
        {"source": "Very very very very very very long sentence.", "target": "Goodbye!"},
    ]