

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
from pathlib import Path
import pandas as pd


@dataclass
class TrainingHistory:
    """
    Stores information about the performance of a model for each epoch
    """
    train_loss: List[float]
    val_loss: List[float]
    val_accuracy: List[float]
    val_mae_class: List[float]
    val_boundary_distance: List[float]


class LinearBoundaryModel(nn.Module):
    """
    Single-layer linear neural network model:
    """

    def __init__(self, n_features: int,init_pred):
        super().__init__()

        # Input (n_features number of inputs) to Score layer (1 output)
        self.linear = nn.Linear(n_features, 1)

        # Weights are initialized as 0
        nn.init.zeros_(self.linear.weight)
        # Bias is set to init_pred because we have boundaries that may affect the model significantly
        nn.init.constant_(self.linear.bias, init_pred)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def validate_bounds(
    bounds: Sequence[Tuple[float, float]],
) -> None:
    """
    In this approach, bounds are part of ordinal analysis

    Through this function, we ensure that the boundaries are set correctly.
    More particularly, we check if the low boundary is indeed lower than the upper boundary.

    Validate the order of the boundaries given upper[k] should be equal to lower[k+1] for each k

    A boundary is given like a tuple -> (lower,upper)
    """

    if len(bounds) < 2:
        raise ValueError("bounds must contain at least two classes.")

    for k, (lower, upper) in enumerate(bounds):
        if lower >= upper:
            raise ValueError(
                f"Invalid bounds for class {k}: lower must be < upper."
            )

        if k < len(bounds) - 1:
            next_lower = bounds[k + 1][0]

            if upper != next_lower:
                raise ValueError(
                    f"Bounds must be contiguous for half-open intervals. "
                    f"Class {k} ends at {upper}, but class {k + 1} "
                    f"starts at {next_lower}."
                )



def labels_to_bounds(
    y: torch.Tensor,
    bounds: Sequence[Tuple[float, float]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Given ordinal values y, the function checks if every value in y is within the range of the possible variable values.
    The values in y should be higher or equal than 0 , and lower or equal to the highest observed ordinal value.

    The function returns the lower and upper boundaries for every ordinal class-value in y.
    """

    y = y.long().reshape(-1)

    if torch.any(y < 0):
        raise ValueError("Class labels must be >= 0.")

    if torch.any(y >= len(bounds)):
        raise ValueError(
            f"Class labels must be between 0 and {len(bounds) - 1}."
        )

    # Build tensors containing all class lower/upper bounds once.
    all_lower = torch.tensor(
        [b[0] for b in bounds],
        dtype=torch.float32,
        device=y.device,
    )

    all_upper = torch.tensor(
        [b[1] for b in bounds],
        dtype=torch.float32,
        device=y.device,
    )

    # Index the correct interval for every observation.
    lower = all_lower[y]
    upper = all_upper[y]

    return lower, upper



def value_to_class(
    value: torch.Tensor,
    bounds: Sequence[Tuple[float, float]],
) -> torch.Tensor:
    """
    Assigns the correct ordinal class-value to a predicted continuous score.
    Lower boundary is closed while upper is open. low<=value<high. Range = [lower, upper).
    """

    value = value.reshape(-1)

    # Lower bounds of classes 1..K are the class thresholds.
    thresholds = torch.tensor(
        [bounds[k][0] for k in range(1, len(bounds))],
        dtype=value.dtype,
        device=value.device,
    )

    return torch.bucketize(
        value,
        thresholds,
        right=True,
    )


def boundary_distance(
    prediction: torch.Tensor,
    y: torch.Tensor,
    bounds: Sequence[Tuple[float, float]],
) -> torch.Tensor:
    """
    Computes the absolute distance (RELU) from a continuous score to the interval of the TRUE class.

    Examples:
    y = 1 , class interval = [1000,2000)
    prediction = 200
    distance = 800
    """

    prediction = prediction.reshape(-1)
    y = y.reshape(-1)

    lower, upper = labels_to_bounds(y, bounds)

    # max(0, lower - prediction)
    # ReLU is used only as a convenient max(0, x), not as a network activation.
    below_distance = torch.relu(lower - prediction)

    # Upper-bound distance is calculated only for finite upper limits.
    finite_upper = torch.isfinite(upper)

    above_distance = torch.zeros_like(prediction)
    above_distance[finite_upper] = torch.relu(
        prediction[finite_upper] - upper[finite_upper]
    )

    return below_distance + above_distance


def boundary_loss(
    prediction: torch.Tensor,
    y: torch.Tensor,
    bounds: Sequence[Tuple[float, float]],
    squared: bool = False,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Calculates MSE or MAE (squared -> given by the user)
    This function is required to adjust loss using class_weights.
    Class weights are used to handle class imbalance, giving higher weight loss to rarer classes.
    Average loss is returned
    """

    distance = boundary_distance(
        prediction,
        y,
        bounds,
    )

    if squared:
        losses = distance ** 2
    else:
        losses = distance

    if class_weights is not None:
        sample_weights = class_weights[y.long()]
        losses = losses * sample_weights

    return losses.mean()


def calculate_class_weights(
    y: torch.Tensor,
    n_classes: int,
) -> torch.Tensor:
    """
    This function calculates class weights based on an array (Tensor in this case).
    Weights are calculated as a function of the number of times each class exists in y (class_counts)
    Weight = 1.0 / class_counts[k] for each k
    Returns the class weights
    """

    y = torch.as_tensor(
        y,
        dtype=torch.long,
    ).reshape(-1)

    if torch.any(y < 0):
        raise ValueError("Target labels cannot contain negative values.")

    if torch.any(y >= n_classes):
        raise ValueError(
            f"Target labels must be between 0 and {n_classes - 1}."
        )

    class_counts = torch.bincount(
        y,
        minlength=n_classes,
    ).float()

    class_weights = torch.zeros_like(class_counts)

    present = class_counts > 0

    class_weights[present] = 1.0 / class_counts[present]

    # Normalize so the mean weight among classes present in training is 1.
    class_weights[present] = (
        class_weights[present]
        / class_weights[present].mean()
    )

    return class_weights


@torch.no_grad()
def evaluate_model(
    model: LinearBoundaryModel,
    x: torch.Tensor,
    y: torch.Tensor,
    bounds: Sequence[Tuple[float, float]],
    squared_loss: bool = False,
    class_weights: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Evaluate the model given x, y, defined boundaries (bounds), and class weights.
    squared_loss is a binary variable denoting whether to use MSE or MAE.

    Returns
    loss: Loss using class weights (MSE or MAE).
    accuracy: Prediction accuracy.
    class_mae: Class MAE without class weight.
    within_one_class_accuracy: Proportion of observations classified with a class distance equal to 1 or 0.
    mean_boundary_distance: Average distance from the desired boundary.

    """

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    x = torch.as_tensor(
        x,
        dtype=torch.float32,
        device=device,
    )

    y = torch.as_tensor(
        y,
        dtype=torch.long,
        device=device,
    ).reshape(-1)

    prediction = model(x)

    class_prediction = value_to_class(
        prediction,
        bounds,
    )

    loss = boundary_loss(
        prediction,
        y,
        bounds=bounds,
        squared=squared_loss,
        class_weights=class_weights,
    )

    # Raw/unweighted distance for interpretability.
    distance = boundary_distance(
        prediction,
        y,
        bounds,
    )

    accuracy = (
        (class_prediction == y)
        .float()
        .mean()
        .item()
    )

    class_mae = (
        (class_prediction.float() - y.float())
        .abs()
        .mean()
        .item()
    )

    within_one = (
        ((class_prediction - y).abs() <= 1)
        .float()
        .mean()
        .item()
    )

    return {
        "loss": float(loss.item()),
        "accuracy": float(accuracy),
        "class_mae": float(class_mae),
        "within_one_class_accuracy": float(within_one),
        "mean_boundary_distance": float(distance.mean().item()),
    }


def train_model(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    bounds: Sequence[Tuple[float, float]],
    x_val: Optional[torch.Tensor] = None,
    y_val: Optional[torch.Tensor] = None,
    *,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    squared_loss: bool = False,
    use_class_weights: bool = True,
    early_stopping: bool = True,
    patience: int = 5,
    min_relative_improvement: float = 0.01,
    device: Optional[str] = None,
    verbose: bool = True,
    init_pred=0
) -> Tuple[LinearBoundaryModel, TrainingHistory]:
    """
    Model training.

    x_train: independent variables used for training
    x_val: independent variables used for validation
    y_train: target variables used for training
    y_val: target variables used for validation

    bounds: Class bounds
    epochs: maximum number of epochs for training
    batch_size: Batch size for back propagation
    lr: learning rate
    squared_loss: Boolean variable denoting whether to use MSE or MAE as loss
    use_class_weights: Boolean variable denoting whether to calculate and use class weights
    early_stopping: Boolean variable denoting whether to use early stopping or not
    min_relative_improvement: The minimum improvement that should be achieved by following epochs to not trigger early stopping
    patience: How many epochs to wait before activating early stopping if min_relative_improvement is not satisfied
    init_pred: Bias defined when the model is initialized. Very important if the boundaries include infinity (-inf , inf)

    Returns the model and the training history (loss, accuracy etc.)
    If early stopping is triggered, the model that was followed by significant improvements is retained and not the last chronologically.
    """

    validate_bounds(bounds)

    # -------------------------------------------------------------------------
    # Input preparation
    # -------------------------------------------------------------------------

    x_train = torch.as_tensor(
        x_train,
        dtype=torch.float32,
    )

    y_train = torch.as_tensor(
        y_train,
        dtype=torch.long,
    ).reshape(-1)

    if x_train.ndim != 2:
        raise ValueError(
            "x_train must have shape [n_samples, n_features]."
        )

    if len(x_train) != len(y_train):
        raise ValueError(
            "x_train and y_train must contain the same number of rows."
        )

    if torch.any(y_train < 0):
        raise ValueError("y_train cannot contain negative class labels.")

    if torch.any(y_train >= len(bounds)):
        raise ValueError(
            f"y_train labels must be between 0 and {len(bounds) - 1}."
        )

    if patience < 1:
        raise ValueError("patience must be >= 1.")

    if min_relative_improvement < 0:
        raise ValueError("min_relative_improvement must be >= 0.")

    # Early stopping makes sense only when validation data exists.
    if early_stopping and (x_val is None or y_val is None):
        raise ValueError(
            "early_stopping=True requires both x_val and y_val."
        )

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_t = torch.device(device)

    # -------------------------------------------------------------------------
    # Optional inverse-frequency class weights
    # -------------------------------------------------------------------------

    class_weights = None

    if use_class_weights:
        class_weights = calculate_class_weights(
            y_train,
            n_classes=len(bounds),
        ).to(device_t)

    # -------------------------------------------------------------------------
    # Model and optimizer
    # -------------------------------------------------------------------------

    model = LinearBoundaryModel(
        n_features=x_train.shape[1],
        init_pred=init_pred
    ).to(device_t)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # -------------------------------------------------------------------------
    # Mini-batch loader
    # -------------------------------------------------------------------------

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
    )

    history = TrainingHistory(
        train_loss=[],
        val_loss=[],
        val_accuracy=[],
        val_mae_class=[],
        val_boundary_distance=[],
    )

    # -------------------------------------------------------------------------
    # Validation data
    # -------------------------------------------------------------------------

    if x_val is not None and y_val is not None:
        x_val = torch.as_tensor(
            x_val,
            dtype=torch.float32,
        )

        y_val = torch.as_tensor(
            y_val,
            dtype=torch.long,
        ).reshape(-1)

        if len(x_val) != len(y_val):
            raise ValueError(
                "x_val and y_val must contain the same number of rows."
            )

        if torch.any(y_val < 0) or torch.any(y_val >= len(bounds)):
            raise ValueError(
                f"y_val labels must be between 0 and {len(bounds) - 1}."
            )

    # -------------------------------------------------------------------------
    # Early-stopping state
    # -------------------------------------------------------------------------

    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = None
    epochs_without_improvement = 0

    # =========================================================================
    # EPOCH LOOP
    # =========================================================================

    for epoch in range(1, epochs + 1):

        model.train()

        running_loss = 0.0
        n_seen = 0

        # ---------------------------------------------------------------------
        # Mini-batch optimization
        # ---------------------------------------------------------------------

        for xb, yb in loader:

            xb = xb.to(device_t)
            yb = yb.to(device_t)

            optimizer.zero_grad()

            # Continuous linear prediction.
            prediction = model(xb)

            # Distance to true class interval, optionally class weighted.
            loss = boundary_loss(
                prediction,
                yb,
                bounds=bounds,
                squared=squared_loss,
                class_weights=class_weights,
            )

            # Compute gradients and update coefficients.
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)

        train_loss = running_loss / max(n_seen, 1)
        history.train_loss.append(train_loss)

        # =====================================================================
        # VALIDATION
        # =====================================================================

        if x_val is not None and y_val is not None:

            metrics = evaluate_model(
                model,
                x_val,
                y_val,
                bounds=bounds,
                squared_loss=squared_loss,
                class_weights=class_weights,
                device=device_t,
            )

            val_loss = metrics["loss"]

            history.val_loss.append(val_loss)
            history.val_accuracy.append(metrics["accuracy"])
            history.val_mae_class.append(metrics["class_mae"])
            history.val_boundary_distance.append(
                metrics["mean_boundary_distance"]
            )

            if verbose and (
                epoch == 1
                or epoch % 1 == 0
                or epoch == epochs
            ):
                print(
                    f"Epoch {epoch:4d}/{epochs} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_acc={metrics['accuracy']:.4f} | "
                    f"val_class_mae={metrics['class_mae']:.4f} | "
                    f"mean_boundary_distance="
                    f"{metrics['mean_boundary_distance']:.4f}"
                )

            # -----------------------------------------------------------------
            # EARLY STOPPING
            # -----------------------------------------------------------------

            if early_stopping:

                # First validation epoch is automatically the current best.
                if best_model_state is None:
                    best_val_loss = val_loss
                    best_model_state = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    epochs_without_improvement = 0

                else:
                    # Relative improvement:
                    #
                    # (old best - new loss) / |old best|
                    #
                    # A small epsilon protects against division by zero.
                    denominator = max(abs(best_val_loss), 1e-12)

                    relative_improvement = (
                        best_val_loss - val_loss
                    ) / denominator

                    if relative_improvement >= min_relative_improvement:
                        # Meaningful improvement -> save model and reset patience.
                        best_val_loss = val_loss
                        best_model_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        epochs_without_improvement = 0

                    else:
                        # Improvement was smaller than the requested percentage,
                        # or validation loss became worse.
                        epochs_without_improvement += 1

                    if epochs_without_improvement >= patience:

                        if verbose:
                            print(
                                f"Early stopping at epoch {epoch}. "
                                f"Best epoch={best_epoch}, "
                                f"best val_loss={best_val_loss:.4f}."
                            )

                        break

        # ---------------------------------------------------------------------
        # Training without validation
        # ---------------------------------------------------------------------

        elif verbose and (
            epoch == 1
            or epoch % 5 == 0
            or epoch == epochs
        ):
            print(
                f"Epoch {epoch:4d}/{epochs} | "
                f"train_loss={train_loss:.4f}"
            )

    # -------------------------------------------------------------------------
    # Restore the best model found by early stopping
    # -------------------------------------------------------------------------

    if early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)

        if verbose:
            print(
                f"Restored model from epoch {best_epoch} "
                f"(val_loss={best_val_loss:.4f})."
            )

    return model, history


@torch.no_grad()
def predict(
    model: LinearBoundaryModel,
    x: torch.Tensor,
    bounds: Sequence[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns predicted scores and the related classes given an input x and defined bounds.
    """

    validate_bounds(bounds)

    device = next(model.parameters()).device

    x = torch.as_tensor(
        x,
        dtype=torch.float32,
        device=device,
    )

    model.eval()

    predicted_value = model(x)

    predicted_class = value_to_class(
        predicted_value,
        bounds,
    )

    return (
        predicted_value.cpu().numpy(),
        predicted_class.cpu().numpy(),
    )


def get_weights(
    model: LinearBoundaryModel,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Returns the direct linear coefficients of the input variables (input -> score) and intercept.
    """

    weights = (
        model.linear.weight
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )

    intercept = float(
        model.linear.bias
        .detach()
        .cpu()
        .item()
    )

    if feature_names is None:
        feature_names = [
            f"x{i}"
            for i in range(len(weights))
        ]

    if len(feature_names) != len(weights):
        raise ValueError(
            "feature_names length must match the number of input features."
        )

    result = {
        name: float(weight)
        for name, weight in zip(feature_names, weights)
    }

    result["intercept"] = intercept

    return result


def fit_and_evaluate(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    bounds: Sequence[Tuple[float, float]],
    feature_names: Optional[Sequence[str]] = None,
    **train_kwargs,
) -> Dict:
    """
    Trains a model using train_model function based on:
    x_train: Independent variables used during model training.
    x_test: Independent variables used during model evaluation.
    y_train: Dependent variable used during model training.
    y_test: Dependent variable used during model evaluation.

    bounds: Predefined bounds given by the user.
    feature_names: Names describing the variables used as independent variables.

    train_kwargs: includes other parameters with the train_model function.

    returns
    model
    model weights
    accuracy metrics
    training history
    predictions (absolute values and classes)
    """

    validate_bounds(bounds)

    # Train model. x_test/y_test act as validation data here.
    model, history = train_model(
        x_train=x_train,
        y_train=y_train,
        bounds=bounds,
        x_val=x_test,
        y_val=y_test,
        **train_kwargs,
    )

    # Recreate training-derived class weights for final evaluation.
    eval_class_weights = None

    if train_kwargs.get("use_class_weights", True):
        eval_class_weights = calculate_class_weights(
            y_train,
            n_classes=len(bounds),
        ).to(next(model.parameters()).device)

    metrics = evaluate_model(
        model,
        x_test,
        y_test,
        bounds=bounds,
        squared_loss=train_kwargs.get("squared_loss", False),
        class_weights=eval_class_weights,
    )

    weights = get_weights(
        model,
        feature_names,
    )

    predicted_value, predicted_class = predict(
        model,
        x_test,
        bounds=bounds,
    )

    return {
        "model": model,
        "weights": weights,
        "metrics": metrics,
        "history": history,
        "predicted_value": predicted_value,
        "predicted_class": predicted_class,
    }


def bootstrap_coefficients(
    X,
    y,
    bounds,
    feature_names,
    init_pred,
    n_boot=100,
    seed=42,

    batch_size=1024,
    lr=0.05,
    squared_loss = False,
    use_class_weights = True,
    early_stopping = True,
    patience = 10,
    min_relative_improvement = 0.01,

    verbose = True,
):
    """
    This function trains a series of models using bootstrap with random resampling for n_boot runs.

    X: matrix containing independent variables
    y: array denoting the class-target variable
    bounds: The defined bounds for each class
    feature_names: Names of the features (independent variables)

    init_pred: Initial bias in the trained models
    n_boot: Number of models and bootstrap runs
    batch_size: Size of the batch used during back propagation
    lr: learning rate
    squared_loss: Whether to use MSE or MAE to calculate loss
    use_class_weights: Whether to use class weights or not to handle class imbalance

    early_stopping: Whether to use early stopping or not
    patience: Number of epochs to wait before activating early stopping
    min_relative_improvement: Minimum improvement that should be exceeded to not trigger early stopping

    Returns coefficients (input -> score) and training history
    """

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    rng = np.random.default_rng(seed)
    estimates = []
    histories= []
    for b in range(n_boot):
        # Sample rows with replacement
        print(f"Bootstrap: {b+1}/{n_boot}")
        idx = rng.integers(0, len(y), size=len(y))

        model, history = train_model(
            x_train=torch.from_numpy(X[idx]),
            y_train=torch.from_numpy(y[idx]),

            x_val=torch.from_numpy(X[idx]),
            y_val=torch.from_numpy(y[idx]),

            bounds=bounds,
            init_pred=init_pred,
            epochs=100,
            batch_size=batch_size,
            lr=lr,
            squared_loss=squared_loss,
            use_class_weights=use_class_weights,
            early_stopping=early_stopping,
            patience = patience,
            min_relative_improvement = min_relative_improvement,

            verbose=verbose,
        )

        estimates.append(get_weights(model, feature_names))
        histories.append(history)

    estimates = pd.DataFrame(estimates)

    return {"Coefs":estimates , "Histories":histories}


def save_bootstrap_results(results, path="bootstrap_results.pkl"):
    """Save bootstrap results, including coefficients, summary and histories."""
    with Path(path).open("wb") as f:
        pickle.dump(results, f)


def load_bootstrap_results(path="bootstrap_results.pkl"):
    """Load bootstrap results. The class TrainingHistory is required."""
    with Path(path).open("rb") as f:
        return pickle.load(f)


if __name__ == "__main__":

    """
    The call used in the paper to model skill frequencies in OJAs with salary and experience levels.
    We trained one model for salary levels and one for experience levels

    The data used are stored in the skill_data_mat variable. This variable is a pandas DataFrame that should be loaded outside the current script.
    skill_data_mat includes multiple variables, were missing values are encoded with the value -1: 
     
    Skill variables (skill_groups): A list of Level-1 ESCO skill groups (columns) for each observation (rows). These attributes correspond to count values.
     
    Control variables (cat_vars, num_vars).
    OJA country, economic_activity1d (Specific sectors), activity_sector (General sector), 
    working_time (work model -> part_time, full_time), Occupation_level (Encoded as 0-3 for 4 different teaching occupations),
    timestamp_scaled_first_active_date (numerical scaled value corresponding to OJA creation date)
    Education_level: Required educational levels ranging from 0-7 (used only for salary levels)
    Experience_level: Required experience levels ranging from 0-7 (used only for salary levels)

    Target variables
    Experience_level: Required experience levels ranging from 0-7
    Salary_level: Salary level mentioned within the OJA ranging from 0-12
     
    The model is built based on all skills (skill_groups) and control variables, 
    both categorical (cat_vars) and numerical (num_vars)
    
    Creates dummy (binary) variables for the categorical control variables.
    
    Excludes observations where the target variables is missing. Missing values were encoded as -1.
    
    Develops 100 models using bootstrap and extracts performance metrics, history,
    and coefficients connecting the input to the score.
    
    The bootstrap results are saved in the computer.
    Then they are loaded to extract bootstrap intervals and evaluate-inspect the consistency of the predictors.
         
    """

    y = "Salary_level" # Experience_level Salary_level


    skill_groups = [
        'information and communication technologies (icts)',
        'business, administration and law',
        'thinking skills and competences',
        'management skills',
        'self-management skills and competences',
        'social and communication skills and competences',
        'generic programmes and qualifications',
        'working with computers',
        'health and welfare',
        'arts and humanities',
        'natural sciences, mathematics and statistics',
        'communication, collaboration and creativity',
        'services',
        'engineering, manufacturing and construction',
        'social sciences, journalism and information',
        'assisting and caring',
        'handling and moving',
        'education',
        'information skills',
        'core skills and competences',
        'working with machinery and specialised equipment',
        'constructing',
        'agriculture, forestry, fisheries and veterinary',
        #'life skills and competences'
    ]




    cat_vars = [
        'Country_code',
        'economic_activity1d',
        'activity_sector',
        'working_time',
        'contract',
        'Occupation_level',
        #'Education_level',
        #'Experience_level',
    ]

    if y=="Salary_level":
        initial_prediction=30000
        cat_vars+=["Education_level","Experience_level"]

        input_bounds = [
            (0.0, 6000.0),
            (6000.0, 12000.0),
            (12000.0, 18000.0),
            (18000.0, 24000.0),
            (24000.0, 30000.0),
            (30000.0, 36000.0),
            (36000.0, 42000.0),
            (42000.0, 48000.0),
            (48000.0, 54000.0),
            (54000.0, 66000.0),
            (66000.0, 78000.0),
            (78000.0, 90000.0),
            (90000.0, float("inf")),
        ]
    else:

        initial_prediction=2

        input_bounds=[
            (float("-inf"),0),
            (0.0,1.0),
            (1.0,2.0),
            (2.0,4.0),
            (4.0,6.0),
            (6.0,8.0),
            (8.0,10.0),
            (10.0,float("inf"))
        ]


    num_vars = skill_groups + ["timestamp_scaled_first_active_date"]

    # Keep only model variables
    Y_df = skill_data_mat[y].to_numpy()

    mask = Y_df != -1

    X_df = skill_data_mat[num_vars + cat_vars].iloc[mask].copy()
    Y_df = Y_df[mask]



    # Temporary check subset
    is_check=False
    if is_check:
        check_size = 10000
        X_df = X_df.iloc[:check_size].copy()
        Y_df = Y_df[:check_size]

    # One-hot encode categorical variables
    X_df = pd.get_dummies(
        X_df,
        columns=cat_vars,
        dtype=int
    )

    bootstrap_coefs=bootstrap_coefficients(
        X=X_df,
        y=Y_df,
        init_pred=initial_prediction,
        bounds=input_bounds,
        feature_names=X_df.columns.tolist(),
        n_boot=100,
        seed=1
    )

    save_bootstrap_results(bootstrap_coefs,f"bootstrap_{y}.pkl")

    #Load results
    res_boots=load_bootstrap_results(f"bootstrap_{y}.pkl")


    # Coefficient uncertainty
    thres = 0.05  # 0.025 0.05
    summary = pd.DataFrame({
        "estimate mean": res_boots['Coefs'].mean(),
        "estimate median": res_boots['Coefs'].median(),
        "ci_lower": res_boots['Coefs'].quantile(thres),
        "ci_upper": res_boots['Coefs'].quantile(1 - thres),
    })
    summary["excludes_zero"] = (summary.ci_lower > 0) | (summary.ci_upper < 0)


