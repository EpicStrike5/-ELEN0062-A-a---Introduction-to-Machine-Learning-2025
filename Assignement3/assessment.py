import numpy as np
from math import copysign as sign
from toy_example import load_from_csv, make_pair_of_players, compute_distance_, write_submission, measure_time
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state
from method1 import detectSides, playerForward

def divideDataset(data):
    split_index = int(data.shape[0] / 4)
    LS = data.loc[np.arange(0, 2 * split_index)]
    VS = data.loc[np.arange(2 * split_index, 3 * split_index)]
    TS = data.loc[np.arange(3 * split_index, data.shape[0])]
    return LS, VS, TS

if __name__ == '__main__':
    prefix = ''

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    X_LS = load_from_csv(prefix+'input_train_set.csv')
    y_LS = load_from_csv(prefix+'output_train_set.csv')

    X_LS, X_VS, X_TS = divideDataset(X_LS)
    y_LS, y_VS, y_TS = divideDataset(y_LS)

    s = detectSides(X_LS)
    # Transform data as pair of players
    # !! This step is only one way of addressing the problem.
    # We strongly recommend to also consider other approaches that the one provided here.

    X_LS_pairs, y_LS_pairs = make_pair_of_players(X_LS, y_LS)
    X_LS_pairs["distance"] = compute_distance_(X_LS_pairs)
    X_LS_pairs["forward"] = playerForward(X_LS_pairs, s)

    X_features = X_LS_pairs[["distance", "same_team", "forward"]]

    s = detectSides(X_VS)
    # Same transformation as LS
    X_VS_pairs, y_VS_pairs = make_pair_of_players(X_VS)
    X_VS_pairs["distance"] = compute_distance_(X_VS_pairs)
    X_VS_pairs["forward"] = playerForward(X_VS_pairs, s)

    X_VS_features = X_VS_pairs[["distance", "same_team", "forward"]]

    hyp = np.arange(5, 20)
    scores = np.zeros(len(hyp))
    for i in hyp:
        print(i)
        # Build the model
        model = DecisionTreeClassifier(max_depth = i)

        with measure_time('Training'):
            print('Training...')
            model.fit(X_features, y_LS_pairs)
        
        # Predict
        scores[i] = model.score(X_VS_features, y_VS_pairs)

    max_depth = hyp[np.argmax(scores)]
    # continuer l'assessment : retrain sur LS + VS puis tester sur TS pour avoir l'estimation de la performance