"""<>"""
import optuna
from objectives import random_forest_regressor
from objectives import random_forest_regressor_multi_parameters

def sklearn_model(x_train, y_train, directions, n_trials):
    """<>"""
    study = optuna.create_study(study_name='optimization', directions=directions)
    study.optimize(lambda trial: random_forest_regressor.objective(trial,
                                            x_train, y_train), n_trials=n_trials)

    best_trial = study.best_trial

    print(f'Accuracy: {best_trial.value}')
    print(f'Best hyperparameters: {best_trial.params}')

    return best_trial.params

def sklearn_model_acc_time(x_train, y_train, directions, n_trials):
    """<>"""
    study = optuna.create_study(study_name='optimization', directions=directions)
    study.optimize(lambda trial: random_forest_regressor_multi_parameters.objective(trial,
                                            x_train, y_train), n_trials=n_trials)

    best_trials = study.best_trials
    best_acc = [trial.values[0] for trial in best_trials]

    return best_trials[best_acc.index(max(best_acc))].params
