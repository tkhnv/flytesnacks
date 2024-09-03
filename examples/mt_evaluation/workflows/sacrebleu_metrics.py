from flytekit import task, ImageSpec

custom_image = ImageSpec(
    packages=["sacrebleu"],
    registry="localhost:30000",
)

@task(container_image=custom_image)
def compute_sacrebleu_metrics(data: list[dict], target_key: str="target", mt_key: str = "mt") -> dict:
    """Compute the length ratio of each segment in a list of dictionaries.
    Args:
        data: A list of segment, has to contain source_key and target_key.
        source_key: The key of the source text in the dictionary.
        target_key: The key of the target text in the dictionary.
        mt_key: The key of the MT in the dictionary.
    """
    import sacrebleu
    refs = [segment[target_key] for segment in data]
    mts = [segment[mt_key] for segment in data]
    bleu = sacrebleu.corpus_bleu(mts, [refs], smooth_method='floor')
    chrf = sacrebleu.corpus_chrf(mts, [refs], beta=3)
    ter = sacrebleu.corpus_ter(mts, [refs], asian_support=True)
    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
        "ter": ter.score
    }
