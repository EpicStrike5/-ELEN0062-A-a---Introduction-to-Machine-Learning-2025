import numpy as np
import pandas as pd
from math import copysign as sign
from toy_example import load_from_csv, make_pair_of_players, compute_distance_, write_submission, measure_time
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import check_random_state

def detectSides(LSx):
    """
    Detect which team plays on which side.
    It is determined by finding the 2 players that have the maximum and minimum position.
    Those players would usually be the goalkeepers so we can deduce from their positions the side their teams are on.
    
    Parameters
    ----------
    LSx: dataframe
        The dataframe containing the data

    Return
    ------
    side: array
        An array of integers either equal to 1 (home team is on the left), -1 (home team is on the right)
        or 0 if both players found are from the same team.
    """
    side = np.zeros(LSx.shape[0] * 22)
    for j in range(LSx.shape[0]):
        max = 0.0
        min = 0.0
        id_max = 0
        id_min = 0
        for i in np.arange(2, 45, 2):
            x = LSx.iloc[j].iloc[i]
            if(x < min):
                min = x
                id_min = int(i/2)
            if(x > max):
                max = x 
                id_max = int(i/2)
        if((id_min <= 11 and id_max <= 11) or (id_min > 11 and id_max > 11)):
            side[j] = 0
            print("Need to improve side detection")
        elif(id_min <= 11):
            side[j * 22: (j + 1) * 22] = 1
        else:
            side[j * 22: (j + 1) * 22] = -1
        print(min, max, id_min, id_max)
    return side
    
def playerForward(pairs, side):
    """
    Detect if the second player of a pair is in front of the sender
    
    Parameters
    ----------
    pairs: data frame
        A data frame with pairs of players containing at least their IDs and their positions along the x-axis
    side: array
        The side as computed by detectSides

    Return
    ------
    forward: integer
        An integer either equal to 1 (the player is in front of the sender), -1 (the player is behind the sender)
        or 0 if both players are from the same team.
    """
    n = pairs.shape[0]
    forward = np.zeros((n,))
    for i in range(n):
        if(pairs["sender"].loc[i] <= 11 and pairs["player_j"].loc[i] <= 11): # both players are from home team
            forward[i] = sign(1, side[i] * (pairs["x_j"].loc[i] - pairs["x_sender"].loc[i]))
        elif(pairs["sender"].loc[i] > 11 and pairs["player_j"].loc[i] > 11): # both players are from away team
            forward[i] = sign(1, (-1) * side[i] * (pairs["x_j"].loc[i] - pairs["x_sender"].loc[i]))
        else: # players are on different teams
            forward[i] = 0
    return forward

def divideDataset(data):
    split_index = int(data.shape[0] / 4)
    LS = data.loc[np.arange(0, 2 * split_index)]
    VS = data.loc[np.arange(2 * split_index, 3 * split_index)]
    TS = data.loc[np.arange(3 * split_index, data.shape[0])]
    return LS, VS, TS

def assessment(input_file, output_file):
    prefix = ''

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    X_LS = load_from_csv(prefix+input_file)
    y_LS = load_from_csv(prefix+output_file)

    X_LS, X_VS, X_TS = divideDataset(X_LS)
    y_LS, y_VS, y_TS = divideDataset(y_LS)
    s = detectSides(X_LS)
    # Transform data as pair of players
    # !! This step is only one way of addressing the problem.
    # We strongly recommend to also consider other approaches that the one provided here.

    X_LS_pairs, y_LS_pairs = make_pair_of_players(X_LS, y_LS)
    X_LS_pairs["distance"] = compute_distance_(X_LS_pairs)
    X_LS_pairs["forward"] = playerForward(X_LS_pairs, s)

    X_LS_features = X_LS_pairs[["distance", "same_team", "forward"]]
    
    print(X_VS, X_VS.iloc[0])
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
            model.fit(X_LS_features, y_LS_pairs)
        
        # Predict
        scores[i - 5] = model.score(X_VS_features, y_VS_pairs)

    max_depth = hyp[np.argmax(scores)]

    # Retrain the model on LS + VS with the optimal hyperparameter
    X_LSVS = pd.concat([X_LS, X_VS])
    y_LSVS = pd.concat([y_LS, y_VS])

    s = detectSides(X_LSVS)

    X_LSVS_pairs, y_LSVS_pairs = make_pair_of_players(X_LSVS, y_LSVS)
    X_LSVS_pairs["distance"] = compute_distance_(X_LSVS_pairs)
    X_LSVS_pairs["forward"] = playerForward(X_LSVS_pairs, s)

    X_LSVS_features = X_LSVS_pairs[["distance", "same_team", "forward"]]

    # Test on TS for model assessment
    s = detectSides(X_TS)
    X_TS_pairs, y_TS_pairs = make_pair_of_players(X_TS, y_TS)
    X_TS_pairs["distance"] = compute_distance_(X_TS_pairs)
    X_TS_pairs["forward"] = playerForward(X_TS_pairs, s)

    X_TS_features = X_TS_pairs[["distance", "same_team", "forward"]]

    # Build the model
    model = DecisionTreeClassifier(max_depth=max_depth)

    with measure_time('Training'):
        print('Training...')
        model.fit(X_LSVS_features, y_LSVS_pairs)
    
    # Predict
    score = model.score(X_TS_features, y_TS_pairs)
    print(score)
    # Could retrain one last time on LS + VS + TS but the dataset is large 
    # so it is not absolutely necessary
    return score

# if __name__ == '__main__':
#     LSx = load_from_csv('input_train_set.csv')
#     s = detectSides(LSx)
#     print(s)

if __name__ == '__main__':
    prefix = ''

    # ------------------------------- Learning ------------------------------- #
    # Load training data
    X_LS = load_from_csv(prefix+'input_train_set.csv')
    y_LS = load_from_csv(prefix+'output_train_set.csv')

    s = detectSides(X_LS)
    # Transform data as pair of players
    # !! This step is only one way of addressing the problem.
    # We strongly recommend to also consider other approaches that the one provided here.

    X_LS_pairs, y_LS_pairs = make_pair_of_players(X_LS, y_LS)
    X_LS_pairs["distance"] = compute_distance_(X_LS_pairs)
    X_LS_pairs["forward"] = playerForward(X_LS_pairs, s)

    X_features = X_LS_pairs[["distance", "same_team", "forward"]]

    # Build the model
    model = DecisionTreeClassifier()

    with measure_time('Training'):
        print('Training...')
        model.fit(X_features, y_LS_pairs)

    # ------------------------------ Prediction ------------------------------ #
    # Load test data
    X_TS = load_from_csv(prefix+'input_test_set.csv')
    print(X_TS.shape)

    s = detectSides(X_TS)
    # Same transformation as LS
    X_TS_pairs, _ = make_pair_of_players(X_TS)
    X_TS_pairs["distance"] = compute_distance_(X_TS_pairs)
    X_TS_pairs["forward"] = playerForward(X_TS_pairs, s)

    X_TS_features = X_TS_pairs[["distance", "same_team", "forward"]]

    # Predict
    y_pred = model.predict_proba(X_TS_features)[:,1]

    # Deriving probas
    probas = y_pred.reshape(X_TS.shape[0], 22)

    # Estimated score of the model
    predicted_score = assessment('input_train_set.csv', 'output_train_set.csv')

    # Making the submission file
    fname = write_submission(probas=probas, estimated_score=predicted_score, file_name="method1_probas")
    print('Submission file "{}" successfully written'.format(fname))

    # -------------------------- Random Prediction -------------------------- #

    random_state = 0
    random_state = check_random_state(random_state)
    predictions = random_state.choice(np.arange(1,23), size=X_TS.shape[0], replace=True)

    fname = write_submission(predictions=predictions, estimated_score=predicted_score, file_name="method1_predictions")
    print('Submission file "{}" successfully written'.format(fname))