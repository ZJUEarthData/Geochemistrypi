from typing import Union

import mlflow


def retrieve_previous_experiment_id(experiment_name: str) -> Union[str, None]:
    """Retrieve the previous experiment with the same name.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment.

    Returns
    -------
    experiment_id : str
        The ID of the experiment.
    """
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment:
        return experiment.experiment_id
    else:
        return None
