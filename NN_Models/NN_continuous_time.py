

from __future__ import annotations

import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingHistory:
    """
    Stores information about the performance of a model for each epoch
    """
    train_loss: list[float]
    val_loss: list[float]
    val_mae: list[float]
    val_rmse: list[float]
    best_epoch: Optional[int] = None


class LinearRegressionModel(nn.Module):
    """
    Single-layer linear neural network model:
    """
    def __init__(self, n_features: int, init_pred: float = 0.0):
        super().__init__()

        # Input (n_features number of inputs) to Score layer (1 output)
        self.linear = nn.Linear(n_features, 1)

        # Weights are initialized as 0
        nn.init.zeros_(self.linear.weight)
        # Bias is set to init_pred
        nn.init.constant_(self.linear.bias, float(init_pred))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def _inputs(X, y=None):
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if X.ndim != 2 or len(X) == 0 or not torch.isfinite(X).all():
        raise ValueError("X must be a nonempty, finite 2D numeric array")
    if y is None:
        return X
    y = torch.as_tensor(np.asarray(y), dtype=torch.float32).reshape(-1)
    if len(X) != len(y) or not torch.isfinite(y).all():
        raise ValueError("y must be finite and have one value per X row")
    return X, y


def _loss(pred, y, loss_type):
    """
    Calculates loss given pred (predictions) and y (true value).
    MSE or MAE is supported
    """
    if loss_type == "mae":
        return (pred - y).abs().mean()
    if loss_type == "mse":
        return ((pred - y) ** 2).mean()
    raise ValueError("loss_type must be 'mae' or 'mse'")


@torch.no_grad()
def evaluate_model(model, X, y, loss_type="mse"):
    """
    evaluates a model based on X (independent variables) and y (target variables)
    MSE (mse) or MAE (mae) can be used
    """

    X, y = _inputs(X, y)
    device = next(model.parameters()).device
    model.eval()
    pred = model(X.to(device))
    y = y.to(device)
    err = pred - y
    return {
        "loss": float(_loss(pred, y, loss_type)),
        "mae": float(err.abs().mean()),
        "rmse": float(err.square().mean().sqrt()),
    }


def train_model(
    x_train, y_train, x_val=None, y_val=None, *, init_pred=0.0,
    epochs=100, batch_size=1024, lr=0.05, weight_decay=0.0,
    loss_type="mse", early_stopping=True, patience=10,
    min_relative_improvement=0.01, device=None, verbose=True,
):
    """

    Model training.

    x_train: independent variables used for training
    x_val: independent variables used for validation
    y_train: target variables used for training
    y_val: target variables used for validation

    epochs: maximum number of epochs for training
    batch_size: Batch size for back propagation
    lr: learning rate
    loss_type: Loss function. Use mse or mae as loss

    early_stopping: Denotes whether early stopping should be triggered
    min_relative_improvement: The minimum improvement that should be achieved by following epochs to not trigger early stopping
    patience: How many epochs to wait before activating early stopping if min_relative_improvement is not satisfied

    Returns the model and the training history (loss, accuracy etc.)
    If early stopping is triggered, the model that was followed by significant improvements is retained and not the last chronologically.

    """

    X, y = _inputs(x_train, y_train)
    if (x_val is None) != (y_val is None):
        raise ValueError("Provide both x_val and y_val, or neither")
    if x_val is not None:
        Xv, yv = _inputs(x_val, y_val)
        if Xv.shape[1] != X.shape[1]:
            raise ValueError("Training and validation feature counts differ")
    elif early_stopping:
        raise ValueError("early_stopping=True requires separate x_val and y_val")
    if epochs < 1 or batch_size < 1 or patience < 1 or min_relative_improvement < 0:
        raise ValueError("Invalid epochs, batch_size, patience, or improvement threshold")
    if loss_type not in ("mae", "mse"):
        raise ValueError("loss_type must be 'mae' or 'mse'")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = LinearRegressionModel(X.shape[1], init_pred).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    history = TrainingHistory([], [], [], [])
    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = _loss(model(xb), yb, loss_type)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        train_loss = total / len(X)
        history.train_loss.append(train_loss)

        if x_val is not None:
            metrics = evaluate_model(model, Xv, yv, loss_type)
            val_loss = metrics["loss"]
            history.val_loss.append(val_loss)
            history.val_mae.append(metrics["mae"])
            history.val_rmse.append(metrics["rmse"])
            if verbose:
                print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.5f} "
                      f"| val_loss={val_loss:.5f} | val_mae={metrics['mae']:.5f} "
                      f"| val_rmse={metrics['rmse']:.5f}")
            if early_stopping:
                improvement = ((best_loss - val_loss) / max(abs(best_loss), 1e-12)
                               if best_state is not None else float("inf"))
                if best_state is None or improvement >= min_relative_improvement:
                    best_loss = val_loss
                    best_state = copy.deepcopy(model.state_dict())
                    history.best_epoch = epoch
                    wait = 0
                else:
                    wait += 1
                if wait >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}; best epoch {history.best_epoch}")
                    break
        elif verbose:
            print(f"Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.5f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, X):
    """
    Predicts a outputs based on a model and an input X (independent variables)
    """

    X = _inputs(X)
    model.eval()
    return model(X.to(next(model.parameters()).device)).cpu().numpy()


def get_weights(model, feature_names: Optional[Sequence[str]] = None):
    """
    Returns model weights connecting the inputs to the score layer
    """


    coefs = model.linear.weight.detach().cpu().numpy().reshape(-1)
    names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(len(coefs))]
    if len(names) != len(coefs) or len(set(names)) != len(names) or "intercept" in names:
        raise ValueError("feature_names must be unique, match X columns, and exclude 'intercept'")
    return dict(zip(names, map(float, coefs))) | {"intercept": float(model.linear.bias.detach().cpu().item())}


def fit_and_evaluate(x_train, y_train, x_test, y_test, feature_names=None, **train_kwargs):
    """
    Returns a model, weights, metrics (performance) and predictions
     based on training inputs and evaluates the mode based on a test-validation dataset

    x_train: Independent variables used in training
    x_test: Independent variables used in evaluation
    y_train: Dependent variables used in training
    y_test: Dependent variables used in evaluation

    train_kwargs: Extra variables considered to train a model
    """
    model, history = train_model(x_train, y_train, x_test, y_test, **train_kwargs)
    return {"model": model, "history": history,
            "weights": get_weights(model, feature_names),
            "metrics": evaluate_model(model, x_test, y_test, train_kwargs.get("loss_type", "mae")),
            "predicted_value": predict(model, x_test)}


def bootstrap_coefficients(
    X, y, feature_names=None, init_pred=0.5, n_boot=100, seed=42, *,
    epochs=100, batch_size=1024, lr=0.05, weight_decay=0.0,
    loss_type="mse", early_stopping=True, patience=10,
    min_relative_improvement=0.01, device=None, verbose=True,
):
    """
    This function trains a series of models using bootstrap with random resampling for n_boot runs.

    X: matrix containing independent variables
    y: array denoting the class-target variable
    feature_names: Names of the features (independent variables)

    init_pred: Initial bias in the trained models
    n_boot: Number of models and bootstrap runs
    epochs: maximum number of epochs for training
    batch_size: Size of the batch used during back propagation
    lr: learning rate
    loss_type: Whether to use MSE (mse) or MAE (mae) to calculate loss

    early_stopping: Whether to use early stopping or not
    patience: Number of epochs to wait before activating early stopping
    min_relative_improvement: Minimum improvement that should be exceeded to not trigger early stopping

    Returns coefficients (input -> score), training history, and bootstrap confidence intervals
    """

    X, y = _inputs(X, y)
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(seed)
    estimates, histories = [], []
    for b in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        xb, yb = X[idx], y[idx]
        if verbose:
            print(f"Bootstrap {b + 1}/{n_boot}")
        model, history = train_model(
            xb, yb, xb if early_stopping else None,
            yb if early_stopping else None,
            init_pred=init_pred, epochs=epochs, batch_size=batch_size,
            lr=lr, weight_decay=weight_decay, loss_type=loss_type,
            early_stopping=early_stopping, patience=patience,
            min_relative_improvement=min_relative_improvement,
            device=device, verbose=verbose,
        )
        estimates.append(get_weights(model, feature_names))
        histories.append(history)
    coefs = pd.DataFrame(estimates)
    summary = pd.DataFrame({
        "bootstrap_mean": coefs.mean(),
        "ci_lower": coefs.quantile(0.025),
        "ci_upper": coefs.quantile(0.975),
    })
    summary["excludes_zero"] = (summary.ci_lower > 0) | (summary.ci_upper < 0)
    return {"Coefs": coefs, "Histories": histories, "Summary": summary}


def save_bootstrap_results(results, path="continuous_bootstrap_results.pkl"):
    """Save coefficients, summary and histories."""
    with Path(path).open("wb") as f:
        pickle.dump(results, f)


def load_bootstrap_results(path="continuous_bootstrap_results.pkl"):
    """Loads the outputs of bootstrap. Need to load the class TrainingHistory first"""
    with Path(path).open("rb") as f:
        return pickle.load(f)


if __name__ == "__main__":

    """
        
        The call used in the paper to model skill frequencies in OJAs with date information.
            
        The data used are stored in the skill_data_mat variable. This variable is a pandas DataFrame that should be loaded outside the current script.
        skill_data_mat includes multiple variables, were missing values are encoded with the value -1. 
         
        Skill variables (skill_groups): A list of Level-1 ESCO skill groups (columns) for each observation (rows). These attributes correspond to count values.
         
        Control variables (cat_vars)
        OJA country, economic_activity1d (Specific sectors), activity_sector (General sector), 
        working_time (work model -> part_time, full_time), Occupation_level (Encoded as 0-3 for 4 different teaching occupations)

        We trained one model for timestamp_scaled_first_active_date.
        This target variable ranges from 0 (earliest date) to 1 (latest date).
        The transformation was completed using min-max scaler.

        The model is built based on all skills (skill_groups) and control variables (cat_vars).

        Dummy (binary) variables are created for the categorical control variables.

        100 models are developed using bootstrap and extracts performance metrics, history,
        and coefficients connecting the input to the score.

        The bootstrap results are saved in the computer.
        Then they are loaded to extract bootstrap intervals and evaluate-inspect the consistency of the predictors.

    """

    y = "timestamp_scaled_first_active_date" # 

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
        'Occupation_level'
    ]

    # Keep only model variables
    Y_df = skill_data_mat[y].to_numpy()
    X_df = skill_data_mat[skill_groups + cat_vars]
    
    # One-hot encode categorical variables
    X_df = pd.get_dummies(
        X_df,
        columns=cat_vars,
        dtype=int
    )


    
    boot = bootstrap_coefficients(
        X=X_df, y=Y_df,feature_names= X_df.columns.tolist(), init_pred=0.5,
        n_boot=100, early_stopping=True,verbose=True
    )
    print(boot["Summary"])
    save_bootstrap_results(boot)


    #Reload
    res_boots=load_bootstrap_results("bootstrap_scaled_timestamps.pkl")


    # Coefficient uncertainty
    thres = 0.05  # 0.025 0.05
    summary = pd.DataFrame({
        "estimate mean": res_boots['Coefs'].mean(),
        "estimate median": res_boots['Coefs'].median(),
        "ci_lower": res_boots['Coefs'].quantile(thres),
        "ci_upper": res_boots['Coefs'].quantile(1 - thres),
    })
    summary["excludes_zero"] = (summary.ci_lower > 0) | (summary.ci_upper < 0)
