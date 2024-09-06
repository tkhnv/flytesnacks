import flytekit
from flytekit import Resources, task, workflow
import flytekit.deck
from flytekit.types.directory import FlyteDirectory

try:
    from .custom_types import EvaluateReturnType
    from .image_specs import transformers_image_spec
except ImportError:
    from image_specs import transformers_image_spec


@task(
    container_image=transformers_image_spec,
    limits=Resources(mem="5G"),
    requests=Resources(mem="4.5G"),
    enable_deck=True,
)
def compare_systems(evaluation_a: EvaluateReturnType, evaluation_b: EvaluateReturnType) -> None:
    from plotly.io import to_html
    import plotly.graph_objects as go

    # Data for the bar plot
    options = ["System A", "System B"]
    scores = [evaluation_a.score, evaluation_b.score]

    # Create the bar plot
    fig = go.Figure(data=[go.Bar(name="Scores", x=options, y=scores, marker=dict(color=["#1f77b4", "#ff7f0e"]))])

    # Update layout to customize the appearance
    fig.update_layout(
        title="MT Score Comparison",
        xaxis_title="Systems",
        yaxis_title="Scores",
        yaxis=dict(range=[0, 100]),  # Set the y-axis range to 0-100
        template="plotly_white",
    )

    deck = flytekit.Deck("pca", flytekit.deck.MarkdownRenderer().to_html("### MT Score Comparison"))
    deck.append(to_html(fig))


@task(
    container_image=transformers_image_spec,
    limits=Resources(mem="5G"),
    requests=Resources(mem="4.5G"),
    enable_deck=True,
)
def dummy_compare_systems() -> None:
    from plotly.io import to_html
    import plotly.graph_objects as go

    # Data for the bar plot
    options = ["System A", "System B"]
    scores = [30.5, 45]

    # Create the bar plot
    fig = go.Figure(data=[go.Bar(name="Scores", x=options, y=scores, marker=dict(color=["#1f77b4", "#ff7f0e"]))])

    # Update layout to customize the appearance
    fig.update_layout(
        title="MT Score Comparison",
        xaxis_title="Systems",
        yaxis_title="Scores",
        yaxis=dict(range=[0, 100]),  # Set the y-axis range to 0-100
        template="plotly_white",
    )

    deck = flytekit.Deck("pca", flytekit.deck.MarkdownRenderer().to_html("### MT Score Comparison"))
    deck.append(to_html(fig))


@workflow
def wf() -> None:
    """Declare workflow called `wf`."""
    return dummy_compare_systems()
