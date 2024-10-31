import optuna
import torch.nn as nn

from tp7 import Model, run, NUM_CLASSES, INPUT_DIM


NB_EPOCHS = 50

def objective(trial):
    HIDDEN_LAYER = trial.suggest_int('HIDDEN_LAYER', 10, 100)
    p_dropout = trial.suggest_float('p_dropout', 0.0, 0.5)
    l1 = trial.suggest_float('l1', 1e-6, 1e-2, log=True)
    l2 = trial.suggest_float('l2', 1e-6, 1e-2, log=True)
    
    model = Model(INPUT_DIM, NUM_CLASSES, HIDDEN_LAYER, p_dropout=p_dropout, normalisation="batchnorm")
    best_test_accuracy = run(NB_EPOCHS, model, l1=l1, l2=l2)
    return best_test_accuracy

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize') 
    study.optimize(objective, n_trials=20)

    print("Best parameters :", study.best_params)
    print("Number of finished trials: ", len(study.trials))
    
    trial = study.best_trial
    print("Best trial:")
    print("* Value: ", trial.value)
    print("* Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

'''
Best parameters : {'HIDDEN_LAYER': 92, 'p_dropout': 0.0302442721315202, 'l1': 4.665662386422075e-06, 'l2': 0.0001437791903614921}
Number of finished trials:  20
Best trial:
Value:  0.9164012670516968
Params:
    HIDDEN_LAYER: 92
    p_dropout: 0.0302442721315202
    l1: 4.665662386422075e-06
    l2: 0.0001437791903614921
'''