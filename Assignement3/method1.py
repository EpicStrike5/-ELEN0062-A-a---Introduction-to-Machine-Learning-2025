import numpy as np
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
    path_inputs: str
        The path to the csv file containing the inputs to load
    delimiter: str (default: ',')
        The csv field delimiter

    Return
    ------
    side: integer
        An integer either equal to 1 (home team is on the left), -1 (home team is on the right)
        or 0 if both players found are from the same team.
    """
    max = 0.0
    min = 0.0
    id_max = 0
    id_min = 0
    for i in np.arange(2, 45, 2):
        x = LSx.loc[0].iloc[i]
        if(x < min):
           min = x
           id_min = int(i/2)
        if(x > max):
            max = x 
            id_max = int(i/2)
    if((id_min <= 11 and id_max <= 11) or (id_min > 11 and id_max > 11)):
        side = 0
        print("Need to improve side detection")
    elif(id_min <= 11):
        side = 1
    else:
        side = -1
    print(min, max, id_min, id_max)
    return side
    
def playerForward(pairs, side):
    """
    Detect if the second player of a pair is in front of the sender
    
    Parameters
    ----------
    pairs: data frame
        A data frame with pairs of players containing at least their IDs and their positions along the x-axis
    side: int
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
            forward[i] = sign(1, side * (pairs["x_j"].loc[i] - pairs["x_sender"].loc[i]))
        elif(pairs["sender"].loc[i] > 11 and pairs["player_j"].loc[i] > 11): # both players are from away team
            forward[i] = sign(1, (-1) * side * (pairs["x_j"].loc[i] - pairs["x_sender"].loc[i]))
        else: # players are on different teams
            forward[i] = 0
        print(i, n)
    return forward

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
    predicted_score = 0.01 # it is quite logical...

    # Making the submission file
    fname = write_submission(probas=probas, estimated_score=predicted_score, file_name="method1_probas")
    print('Submission file "{}" successfully written'.format(fname))

    # -------------------------- Random Prediction -------------------------- #

    random_state = 0
    random_state = check_random_state(random_state)
    predictions = random_state.choice(np.arange(1,23), size=X_TS.shape[0], replace=True)

    fname = write_submission(predictions=predictions, estimated_score=predicted_score, file_name="method1_predictions")
    print('Submission file "{}" successfully written'.format(fname))