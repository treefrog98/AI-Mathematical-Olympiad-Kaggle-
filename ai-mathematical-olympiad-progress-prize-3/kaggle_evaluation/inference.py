import os

import kaggle_evaluation.aimo_3_inference_server
import pandas as pd
import polars as pl


class Model:
    """A dummy model."""

    def __init__(self):
        self._model = None

    def load(self):
        """Simulate model loading."""
        print("Loading model...")
        # Just return a "model" that always answers with 0
        return lambda problem: 0

    def predict(self, problem: str):
        # Employ lazy loading: load model on the first model.predict call
        if self._model is None:
            self._model = self.load()
        return self._model(problem)


model = Model()


# Replace this function with your inference code.
# The function should return a single integer between 0 and 99999, inclusive.
def predict(id_: pl.Series, problem: pl.Series) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction."""
    # Unpack values
    id_ = id_.item(0)
    problem_text: str = problem.item(0)
    # Make a prediction
    # The model is loaded on the first call
    prediction = model.predict(problem_text)
    return pl.DataFrame({'id': id_, 'answer': prediction})


inference_server = kaggle_evaluation.aimo_3_inference_server.AIMO3InferenceServer(
    predict
)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # You MUST call this within 15 minutes of the script starting. This is to
    # ensure a "fast fail" in case a bug prevents the inference server from starting.
    # Do anything that might take a long time (like model loading) in the predict
    # function, which has no time limit.
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        ('/kaggle/input/ai-mathematical-olympiad-progress-prize-3/test.csv',)
    )