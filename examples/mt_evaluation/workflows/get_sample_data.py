from flytekit import task
import flytekit


@task()
def get_sample_data() -> list[dict]:
    """Get sample data for the mt_evaluation example."""
    return [
        {"source": "Hello, world!", "target": "Bonjour, le monde!", "mt": "Bonjour, le world!"},
        {"source": "This is a test.", "target": "C'est un test.", "mt": "C'est un essay."},
        {"source": "Goodbye!", "target": "Au revoir!", "mt": "A bientot!"},
    ]