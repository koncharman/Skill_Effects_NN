

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
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingHistory:
    """
    Stores information about the performance of a model for each epoch
    """

    train_loss: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    val_class_mae: list[float]
    best_epoch: Optional[int] = None


class CORALOrdinalModel(nn.Module):
    """
    An architecture similar to CORAL for consistent ordinal ranking between ordered target values.

    The implemented architecture receives a series of inputs, computes a score and this score is compared
    to thresholds to provide logits.
    These thresholds denote the probability of an observation being higher than a specific ordinal value.

    The thresholds are computed using trainable parameters, while each threshold is calculated
    based on the previous plus a strictly positive value.

    The consistency of the thresholds is ensured by using:
    F.softplus(self.raw_gaps) + 1e-6
    torch.cumsum(gaps, dim=0)

    The true ordinal value-class is transformed to the number of classes exceded using a vector of binary values.

    For example, given a range of classes 0-5, an observed value equal to 2 exceeds two classes and thresholds,
    i.e., the values 0 and 1, but is not higher than the other 3.
    The transformed ordinal class-value would be [1,1,0,0,0].

    """

    def __init__(self, n_features, n_classes, init_score=0.0):
        super().__init__()
        if n_features < 1 or n_classes < 2:
            raise ValueError('Need >=1 feature and >=2 ordered classes')
        self.n_classes = n_classes
        self.linear = nn.Linear(n_features, 1, bias=False)
        nn.init.zeros_(self.linear.weight)
        self.first_threshold = nn.Parameter(torch.tensor(-float(init_score)))
        self.raw_gaps = nn.Parameter(torch.zeros(n_classes - 2))

    def thresholds(self):
        first = self.first_threshold.reshape(1)
        if self.raw_gaps.numel() == 0:
            return first
        gaps = F.softplus(self.raw_gaps) + 1e-6
        return torch.cat((first, first + torch.cumsum(gaps, dim=0)))

    def forward(self, x):
        score = self.linear(x)
        return score - self.thresholds()  # raw logits; do NOT sigmoid before BCEWithLogits


def _X(X):
    """
    Transforms an input matrix X to a torch array
    """

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    if X.ndim != 2 or len(X) == 0 or not bool(torch.isfinite(X).all()):
        raise ValueError('X must be nonempty, finite, numeric 2D data')
    return X


def _labels(y, class_labels):
    """
    Makes sure that class labels (class_labels) contains all possible values in y (target outputs used during training).
    Transforms the labels (class_labels) into numerical ordinal values.
    """

    labels = list(class_labels)
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError('class_labels must contain >=2 unique LOW-to-HIGH labels')
    mapping = {v: i for i, v in enumerate(labels)}
    try:
        indices = [mapping[v] for v in np.asarray(y).reshape(-1)]
    except (KeyError, TypeError) as exc:
        raise ValueError('y contains labels not in class_labels') from exc
    return torch.tensor(indices, dtype=torch.long)


def ordinal_targets(y_indices, n_classes):
    """
    Transforms ordinal values (y_indices) to binary vectors denoting the thresholds passed by each value.
    The number of thresholds is defined as n_classes-1.
    """

    return (y_indices[:, None] > torch.arange(n_classes - 1, device=y_indices.device)).float()


def class_weights_from_y(y_indices, n_classes):
    """
    This function calculates class weights based on an array (y_indices -> Tensor in this case).
    Weights are calculated as a function of the number of times each class exists in y (counts)
    Weight = 1.0 / counts[k] for each k
    Returns the class weights
    """

    counts = torch.bincount(y_indices.cpu(), minlength=n_classes).float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    weights *= len(y_indices) / (counts * weights).sum().clamp_min(1)
    return weights


def _bce(logits, y_indices, n_classes, class_weights=None):
    """
    Function required to calculate weighted average binary cross entropy (BCE) to handle class imbalance
    according to class weights.

    logits: predicted probabilities associating the score layer to the thresholds,
    i.e., probability that the score exceeds the thresholds

    y_indices: True ordinal values
    """

    targets = ordinal_targets(y_indices, n_classes)
    per_row = F.binary_cross_entropy_with_logits(logits, targets, reduction='none').sum(dim=1)
    if class_weights is not None:
        per_row = per_row * class_weights[y_indices]
    return per_row.mean()


@torch.no_grad()
def predict_score(model, X):
    """
    Predict a score given a neural network model and input X (independent variables).
    """

    model.eval()
    return model.linear(_X(X).to(next(model.parameters()).device)).squeeze(1).cpu().numpy()


@torch.no_grad()
def predict_cumulative_proba(model, X):
    """
    Predicts probabilities of exceeding the model's trained thresholds given X (independent variables).
    """

    model.eval()
    return torch.sigmoid(model(_X(X).to(next(model.parameters()).device))).cpu().numpy()


def predict_proba(model, X):
    """
    Predicts the probability of the observations belonging to each class,
    given the predicted probabilities of the observation exceeding the model thresholds.

    This function is used in case the user selects to predict the most probable class.
    """

    gt = predict_cumulative_proba(model, X)
    return np.concatenate((1 - gt[:, :1], gt[:, :-1] - gt[:, 1:], gt[:, -1:]), axis=1)


def predict(model, X, class_labels=None, method='threshold'):
    """
    Predicts outputs given X (independent variables)

    The user can select to predict the class using the 'method' parameter where:
    'threshold': Assign the class based on the probabilities
    of exceeding the thresholds that are higher than 50%
    'argmax': Assign the most probable class according to the extracted probabilities

    Returns the labels.
    """

    if method == 'threshold':
        indices = (predict_cumulative_proba(model, X) > 0.5).sum(axis=1)
    elif method == 'argmax':
        indices = predict_proba(model, X).argmax(axis=1)
    else:
        raise ValueError("method must be 'threshold' or 'argmax'")
    if class_labels is None:
        return indices
    labels = np.asarray(class_labels)
    if len(labels) != model.n_classes:
        raise ValueError('class_labels count mismatch')
    return labels[indices]


@torch.no_grad()
def evaluate_model(model, X, y, class_labels, class_weights=None):
    """
    Evaluate the performance of the model given X (independent variables) y (dependent variable)
    and class_weights

    Note that class prediction is made using the 'threshold' mode from predict() function hardcoded.

    Returns loss, accuracy, and MAE
    """

    X, y = _X(X), _labels(y, class_labels)
    if len(X) != len(y) or X.shape[1] != model.linear.in_features or len(class_labels) != model.n_classes:
        raise ValueError('Evaluation data shape mismatch')
    model.eval()
    device = next(model.parameters()).device
    logits = model(X.to(device))
    y_device = y.to(device)
    weights = class_weights.to(device) if class_weights is not None else None
    pred = (torch.sigmoid(logits) > 0.5).sum(dim=1)
    return {'loss': float(_bce(logits, y_device, model.n_classes, weights)),
            'accuracy': float((pred == y_device).float().mean()),
            'class_mae': float((pred - y_device).abs().float().mean())}


def train_model(x_train, y_train, x_val=None, y_val=None, *, class_labels,
                init_score=0.0, use_class_weights=True, epochs=100, batch_size=1024,
                lr=0.05, weight_decay=0.0, early_stopping=True, patience=10,
                min_relative_improvement=0.01, device=None, verbose=True):
    """
        Model training.

        x_train: independent variables used for training
        x_val: independent variables used for validation
        y_train: target variables used for training
        y_val: target variables used for validation

        class_labels: True labels of target variable
        init_score: Bias defined when the model is initialized
        use_class_weights: Boolean variable denoting whether to calculate and use class weights

        epochs: maximum number of epochs for training
        batch_size: Batch size for back propagation
        lr: learning rate

        early_stopping: Boolean variable denoting whether to use early stopping or not
        min_relative_improvement: The minimum improvement that should be achieved by following epochs to not trigger early stopping
        patience: How many epochs to wait before activating early stopping if min_relative_improvement is not satisfied

        Returns the model and the training history (loss, accuracy etc.)
        If early stopping is triggered, the model that was followed by significant improvements is retained and not the last chronologically.
        """

    X, y = _X(x_train), _labels(y_train, class_labels)
    if len(X) != len(y):
        raise ValueError('Training X/y row count mismatch')
    if (x_val is None) != (y_val is None):
        raise ValueError('Provide both validation arrays or neither')
    if x_val is not None:
        Xv, yv = _X(x_val), _labels(y_val, class_labels)
        if len(Xv) != len(yv) or Xv.shape[1] != X.shape[1]:
            raise ValueError('Validation shape mismatch')
    elif early_stopping:
        raise ValueError('Early stopping requires validation arrays')
    if epochs < 1 or batch_size < 1 or patience < 1 or lr <= 0 or min_relative_improvement < 0:
        raise ValueError('Invalid training hyperparameters')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model = CORALOrdinalModel(X.shape[1], len(class_labels), init_score).to(device)
    class_weights = class_weights_from_y(y, model.n_classes).to(device) if use_class_weights else None
    model.class_weights_ = class_weights.detach().cpu() if class_weights is not None else None
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    history = TrainingHistory([], [], [], [])
    best_loss, patience_reference, best_state, wait = float('inf'), float('inf'), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = _bce(model(xb), yb, model.n_classes, class_weights)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(xb)
        train_loss = total / len(X)
        history.train_loss.append(train_loss)
        if x_val is not None:
            metrics = evaluate_model(model, Xv, yv, class_labels, class_weights)
            val_loss = metrics['loss']
            history.val_loss.append(val_loss)
            history.val_accuracy.append(metrics['accuracy'])
            history.val_class_mae.append(metrics['class_mae'])
            if verbose:
                print(f'Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} '
                      f"| val_acc={metrics['accuracy']:.4f} | val_class_mae={metrics['class_mae']:.4f}")
            if early_stopping:
                if val_loss < best_loss:
                    best_loss, best_state, history.best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch
                improvement = ((patience_reference - val_loss) / max(abs(patience_reference), 1e-12)
                               if np.isfinite(patience_reference) else float('inf'))
                if improvement >= min_relative_improvement:
                    patience_reference, wait = val_loss, 0
                else:
                    wait += 1
                if wait >= patience:
                    if verbose:
                        print(f'Early stopping at epoch {epoch}; best epoch {history.best_epoch}')
                    break
        elif verbose:
            print(f'Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.5f}')
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def get_weights(model, feature_names: Optional[Sequence[str]] = None):
    """
    Returns the latent thresholds and all the weights calculated during trainiing.
    input -> score
    score -> outputs based on thresholds
    """

    coef = model.linear.weight.detach().cpu().numpy().reshape(-1)
    names = list(feature_names) if feature_names is not None else [f'x{i}' for i in range(len(coef))]
    if len(names) != len(coef) or len(set(names)) != len(names):
        raise ValueError('feature_names must be unique and match feature count')
    thresholds = model.thresholds().detach().cpu().numpy()
    return {'input_to_score': dict(zip(names, map(float, coef))),
            'score_to_output': {f'above_class_{k}': {'weight': 1.0, 'bias': -float(t)}
                                for k, t in enumerate(thresholds)},
            'latent_thresholds': {f'above_class_{k}': float(t) for k, t in enumerate(thresholds)}}


def _flat_weights(model, feature_names=None):
    """
    Concatenates the model weights and raw thresholds to a single array using the feature names (independent variables)
    """

    w = get_weights(model, feature_names)
    flat = {f'input_to_score::{name}': value for name, value in w['input_to_score'].items()}
    for name, pars in w['score_to_output'].items():
        flat[f'score_to_output::{name}::weight'] = pars['weight']
        flat[f'score_to_output::{name}::bias'] = pars['bias']
    for name, value in w['latent_thresholds'].items():
        flat[f'latent_threshold::{name}'] = value
    return flat


def fit_and_evaluate(x_train, y_train, x_val, y_val, feature_names=None, **train_kwargs):
    """
    Trains a model using train_model function based on:
    x_train: Independent variables used during model training.
    x_val: Independent variables used during model evaluation.
    y_train: Dependent variable used during model training.
    y_val: Dependent variable used during model evaluation.

    feature_names: Names describing the variables used as independent variables.

    train_kwargs: includes other parameters with the train_model function.

    returns
    model
    model weights
    accuracy metrics
    training history
    predictions (score, predicted class, and class probabilities)

    """


    labels = train_kwargs['class_labels']
    model, history = train_model(x_train, y_train, x_val, y_val, **train_kwargs)
    return {'model': model, 'history': history, 'weights': get_weights(model, feature_names),
            'metrics': evaluate_model(model, x_val, y_val, labels, model.class_weights_),
            'predicted_class': predict(model, x_val, labels),
            'scores': predict_score(model, x_val), 'class_probabilities': predict_proba(model, x_val)}


def bootstrap_coefficients(X, y, feature_names=None, *, class_labels, n_boot=100, seed=1,
                           init_score=0.0, use_class_weights=True, epochs=100, batch_size=1024,
                           lr=0.05, weight_decay=0.0, early_stopping=True, patience=10,
                           min_relative_improvement=0.01, device=None, verbose=True):
    """
        This function trains a series of models using bootstrap with random resampling for n_boot runs.

        X: matrix containing independent variables
        y: array denoting the class-target variable
        feature_names: Names of the features (independent variables)

        n_boot: Number of models and bootstrap runs
        init_score: Initial bias in the trained models
        use_class_weights: Whether to use class weights or not to handle class imbalance
        epochs: maximum number of epochs for training
        batch_size: Size of the batch used during back propagation
        lr: learning rate

        early_stopping: Whether to use early stopping or not
        patience: Number of epochs to wait before activating early stopping
        min_relative_improvement: Minimum improvement that should be exceeded to not trigger early stopping

        Returns coefficients (input -> score, score -> thresholds, thresholds), training history and summary on confidence intervals
        """

    X, y = _X(X), _labels(y, class_labels)
    if len(X) != len(y) or n_boot < 1:
        raise ValueError('X/y row count mismatch or n_boot < 1')
    rng = np.random.default_rng(seed)
    estimates, histories = [], []
    for b in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        xb, yb = X[idx], y[idx]
        if verbose:
            print(f'Bootstrap {b + 1}/{n_boot}')
        model, history = train_model(
            xb, yb, xb if early_stopping else None, yb if early_stopping else None,
            class_labels=class_labels, init_score=init_score, use_class_weights=use_class_weights,
            epochs=epochs, batch_size=batch_size, lr=lr, weight_decay=weight_decay,
            early_stopping=early_stopping, patience=patience,
            min_relative_improvement=min_relative_improvement, device=device, verbose=verbose)
        estimates.append(_flat_weights(model, feature_names))
        histories.append(history)
    coefs = pd.DataFrame(estimates)
    summary = pd.DataFrame({'bootstrap_mean': coefs.mean(), 'ci_lower': coefs.quantile(0.025),
                            'ci_upper': coefs.quantile(0.975)})
    summary['excludes_zero'] = ((summary.ci_lower > 0) | (summary.ci_upper < 0)).where(
        summary.index.str.startswith('input_to_score::'), pd.NA)
    return {'Coefs': coefs, 'Histories': histories, 'Summary': summary}


def save_bootstrap_results(results, path='coral_bootstrap_results.pkl'):
    """Save coefficients, summary and histories."""

    with Path(path).open('wb') as f:
        pickle.dump(results, f)


def load_bootstrap_results(path='coral_bootstrap_results.pkl'):
    """Loads the outputs of bootstrap. Need to load the class TrainingHistory first"""

    with Path(path).open('rb') as f:
        return pickle.load(f)  # only load trusted pickle files


if __name__ == '__main__':
    """
     The call used in the paper to model skill frequencies in OJAs with education levels.
     We trained one model for Education_level

     The data used are stored in the skill_data_mat variable. This variable is a pandas DataFrame that should be loaded outside the current script.
     skill_data_mat includes multiple variables, were missing values are encoded with the value -1: 
     
     Skill variables (skill_groups): A list of Level-1 ESCO skill groups (columns) for each observation (rows). These attributes correspond to count values.
     
     Control variables (cat_vars, num_vars)
     OJA country, economic_activity1d (Specific sectors), activity_sector (General sector), 
     working_time (work model -> part_time, full_time), Occupation_level (Encoded as 0-3 for 4 different teaching occupations),
     timestamp_scaled_first_active_date (numerical scaled value corresponding to OJA creation date)
     
     Target variables
     Education_level: Required educational levels ranging from 0-7
     
    
     The model is built based on all skills (skill_groups) and control variables, 
     both categorical (cat_vars) and numerical (num_vars)

     Creates dummy (binary) variables for the categorical control variables.

     Excludes observations where the target variables is missing. Missing values were encoded as -1.

     Develops 100 models using bootstrap and extracts performance metrics, history,
     and coefficients connecting the input to the score and to the thresholds.

     The bootstrap results are saved in the computer.
     Then they are loaded to extract bootstrap intervals and evaluate-inspect the consistency of the predictors.

     """

    y = "Education_level"  #

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

    num_vars="timestamp_scaled_first_active_date"

    # Keep only model variables
    Y_df = skill_data_mat[y].to_numpy()
    mask=(Y_df!=-1)
    X_df = skill_data_mat[skill_groups + cat_vars + [num_vars]].iloc[mask]
    Y_df=Y_df[mask]

    # One-hot encode categorical variables
    X_df = pd.get_dummies(
        X_df,
        columns=cat_vars,
        dtype=int
    )

    class_labels=list(range(min(Y_df),max(Y_df)+1))

    boot = bootstrap_coefficients(X_df, Y_df,feature_names=X_df.columns.tolist(), class_labels=class_labels,
                                  n_boot=100,epochs=100, patience=10, verbose=True)

    save_bootstrap_results(boot,'coral_bootstrap_results_education_level.pkl')

    #Load results
    res_boots = load_bootstrap_results("coral_bootstrap_results_education_level.pkl")

    # Coefficient uncertainty
    thres = 0.05  # 0.025 0.05
    summary = pd.DataFrame({
        "estimate mean": res_boots['Coefs'].mean(),
        "estimate median": res_boots['Coefs'].median(),
        "ci_lower": res_boots['Coefs'].quantile(thres),
        "ci_upper": res_boots['Coefs'].quantile(1 - thres),
    })
    summary["excludes_zero"] = (summary.ci_lower > 0) | (summary.ci_upper < 0)
