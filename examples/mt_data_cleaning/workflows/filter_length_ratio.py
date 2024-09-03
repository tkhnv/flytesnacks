from flytekit import task


@task()
def compute_length_ratio(data: list[dict], max_ratio: float = 2.0, max_length: int = 200, source_key: str = "source", target_key: str="target") -> list[dict]:
    """Compute the length ratio of each segment in a list of dictionaries.
    Args:
        data: A list of segment, has to contain source_key and target_key.
        max_ratio: The maximum allowed length ratio of a segment.
        max_length: The maximum allowed length of a segment.
        source_key: The key of the source text in the dictionary.
        target_key: The key of the target text in the dictionary.

    """
    for segment in data:
        source = segment[source_key]
        target = segment[target_key]
        if len(source) / len(target) > max_ratio or len(target) / len(source) > max_ratio or len(source) > max_length or len(target) > max_length:
            if segment.get("valid") is None or segment["valid"]:
                segment["valid"] = False
                segment['checks_failed'] = segment.get('checks_failed', []) + ["length_ratio"]
        else:
            segment["valid"] = True
    return data

