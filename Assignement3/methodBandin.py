import os
import time
import datetime
from contextlib import contextmanager
from math import copysign as sign, acos, sqrt, pi, degrees

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier 
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, confusion_matrix

# =============================================================================
# CONFIGURATION
# =============================================================================

GRAPH_OUTPUT_DIR = "Graphs" # Local folder for graphs

# Pitch Dimensions (Centimeters)
PITCH_X_MAX = 5250.0
PITCH_Y_MAX = 3400.0
GOAL_X = 5250.0
GATE_WIDTH = 732.0 

# =============================================================================
# GEOMETRY & PHYSICS HELPERS
# =============================================================================

@contextmanager
def measure_time(label):
    """
    Execution Timer Context Manager
    
    Inputs: 
        - label: string (Description of the block being timed)
    Outputs: 
        - None (Yields control, prints duration to stdout)

    Measures and prints the wall-clock execution time of the enclosed code block for performance profiling.
    """
    start = time.time()
    yield
    elapsed = datetime.timedelta(seconds=time.time() - start)
    print(f'Duration of [{label}]: {elapsed}')

def dist_point_to_segment(p, a, b):
    """
    Geometric Distance Calculator
    
    Inputs: 
        - p: numpy array shape (2,) (Point coordinates)
        - a: numpy array shape (2,) (Segment start coordinates)
        - b: numpy array shape (2,) (Segment end coordinates)
    Outputs: 
        - distance: float (Euclidean distance)

    Calculates the shortest perpendicular distance from a point (opponent) to a line segment (passing lane).
    """
    ab = b - a
    ap = p - a
    len_sq = np.dot(ab, ab)
    if len_sq == 0: return np.linalg.norm(ap)
    t = np.dot(ap, ab) / len_sq
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)

def get_angle_between_vectors(v1, v2):
    """
    Vector Angle Calculator
    
    Inputs: 
        - v1: numpy array shape (2,) (Vector 1)
        - v2: numpy array shape (2,) (Vector 2)
    Outputs: 
        - angle: float (Radians)

    Computes the angle between the pass trajectory and an opponent's position vector relative to the sender.
    """
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0: return 0.0
    cos_angle = np.dot(v1, v2) / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)

def softmax(x):
    """
    Probability Distribution Normalizer
    
    Inputs: 
        - x: numpy array shape (N_samples, 22) (Raw model scores)
    Outputs: 
        - probabilities: numpy array shape (N_samples, 22) (Summing to 1 per row)

    Converts raw classifier scores into a calibrated probability distribution across the 22 potential receivers.
    """
    # Subtract max for numerical stability
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# =============================================================================
# SENDER FEATURE HELPERS
# =============================================================================

def calc_s_zones(sender_pos, side):
    """
    Sender Pitch Zone Classifier
    
    Inputs: 
        - sender_pos: numpy array shape (2,)
        - side: integer (1 or -1, playing direction)
    Outputs: 
        - s_back, s_mid, s_front: 3 separate integers (0 or 1)

    Categorizes the ball carrier's position into defensive, middle, or attacking thirds relative to their playing direction.
    """
    sender_x_norm = sender_pos[0] * side
    # Thresholds converted to CM (17.5m -> 1750cm)
    s_back = 1 if sender_x_norm < -1750 else 0 
    s_mid = 1 if -1750 <= sender_x_norm <= 1750 else 0 
    s_front = 1 if sender_x_norm > 1750 else 0 
    return s_back, s_mid, s_front

def calc_s_landmark_dists(sender_pos, offense_gate, defense_gate):
    """
    Sender Landmark Distance Calculator
    
    Inputs: 
        - sender_pos: numpy array shape (2,)
        - offense_gate: numpy array shape (2,) (Target goal center)
        - defense_gate: numpy array shape (2,) (Own goal center)
    Outputs: 
        - s_to_off_gate: float
        - s_to_def_gate: float
        - s_to_center: float

    Measures the sender's proximity to critical pitch locations to assess attacking threat or defensive danger.
    """
    s_to_off_gate = np.linalg.norm(sender_pos - offense_gate) # Distance to the opponent's goal
    s_to_def_gate = np.linalg.norm(sender_pos - defense_gate) # Distance to own goal
    s_to_center = np.linalg.norm(sender_pos) # Distance to the center of the pitch (0,0)
    return s_to_off_gate, s_to_def_gate, s_to_center

def calc_s_player_dists(sender_pos, teammates_pos, opponents_pos):
    """
    Sender Immediate Context Calculator
    
    Inputs: 
        - sender_pos: numpy array shape (2,)
        - teammates_pos: numpy array shape (10, 2)
        - opponents_pos: numpy array shape (11, 2)
    Outputs: 
        - s_closest_tm: float (Min dist to teammate)
        - s_closest_3_tm: float (Avg dist to 3 closest teammates)
        - s_closest_opp: float (Min dist to opponent)
        - s_closest_3_opp: float (Avg dist to 3 closest opponents)

    Quantifies the immediate pressure (opponents) and support (teammates) surrounding the ball carrier.
    """
    tm_dists = np.linalg.norm(teammates_pos - sender_pos, axis=1)
    # Epsilon converted to CM (0.001m -> 0.1cm)
    tm_dists_excl_sender = tm_dists[tm_dists > 0.1] 
    opp_dists = np.linalg.norm(opponents_pos - sender_pos, axis=1)
    
    s_closest_tm = np.min(tm_dists_excl_sender) if len(tm_dists_excl_sender) > 0 else 10000 # Distance to nearest teammate
    s_closest_3_tm = np.mean(np.sort(tm_dists_excl_sender)[:3]) if len(tm_dists_excl_sender) >= 3 else s_closest_tm # Avg distance to 3 closest teammates
    
    s_closest_opp = np.min(opp_dists) if len(opp_dists) > 0 else 10000 # Distance to nearest opponent
    s_closest_3_opp = np.mean(np.sort(opp_dists)[:3]) if len(opp_dists) >= 3 else s_closest_opp # Avg distance to 3 closest opponents
    
    return s_closest_tm, s_closest_3_tm, s_closest_opp, s_closest_3_opp

def calc_s_ranks(sender_pos, teammates_pos, opponents_pos, offense_gate):
    """
    Sender Relative Ranking Calculator
    
    Inputs: 
        - sender_pos: numpy array shape (2,)
        - teammates_pos: numpy array shape (10, 2)
        - opponents_pos: numpy array shape (11, 2)
        - offense_gate: numpy array shape (2,)
    Outputs: 
        - rank_s_off_gate_tm: int (Rank among team to goal)
        - rank_s_off_gate_opp: int (Rank among opponents to goal)
        - rank_s_top_tm: int (Rank among team to top line)
        - rank_s_top_opp: int (Rank among opponents to top line)

    Determines the sender's hierarchical standing (e.g., 'most advanced player') compared to all other players on the pitch.
    """
    s_to_off_gate = np.linalg.norm(sender_pos - offense_gate)
    tm_to_off_gate = np.linalg.norm(teammates_pos - offense_gate, axis=1)
    opp_to_off_gate = np.linalg.norm(opponents_pos - offense_gate, axis=1)
    
    rank_s_off_gate_tm = np.sum(tm_to_off_gate < s_to_off_gate) + 1 # Rank among teammates to goal
    rank_s_off_gate_opp = np.sum(opp_to_off_gate < s_to_off_gate) + 1 # Rank among opponents to goal
    
    top_sideline_y = PITCH_Y_MAX
    s_to_top = abs(sender_pos[1] - top_sideline_y)
    tm_to_top = np.abs(teammates_pos[:, 1] - top_sideline_y)
    opp_to_top = np.abs(opponents_pos[:, 1] - top_sideline_y)
    
    rank_s_top_tm = np.sum(tm_to_top < s_to_top) + 1 # Rank among teammates to top sideline
    rank_s_top_opp = np.sum(opp_to_top < s_to_top) + 1 # Rank among opponents to top sideline
    
    return rank_s_off_gate_tm, rank_s_off_gate_opp, rank_s_top_tm, rank_s_top_opp

def calc_team_geometry(teammates_pos, offense_gate, defense_gate):
    """
    Team Shape / Macro-Geometry Calculator
    
    Inputs: 
        - teammates_pos: numpy array shape (10, 2)
        - offense_gate: numpy array shape (2,)
        - defense_gate: numpy array shape (2,)
    Outputs: 
        - st_closest_off_line: float (Min dist to attack line)
        - st_closest_def_no_gk: float (Deepest field player dist)
        - st_closest_top: float (Min dist to top line)
        - st_closest_bottom: float (Min dist to bottom line)
        - st_median_off_line: float (Median dist to attack line)
        - st_median_top: float (Median dist to top line)

    Captures the macro-structural properties of the team, such as team width, defensive depth, and median field position.
    """
    tm_to_off_line = np.abs(teammates_pos[:, 0] - offense_gate[0])
    tm_to_def_line = np.abs(teammates_pos[:, 0] - defense_gate[0])
    tm_to_top = np.abs(teammates_pos[:, 1] - PITCH_Y_MAX)
    
    st_closest_off_line = np.min(tm_to_off_line) # Closest teammate to opponent goal line
    st_median_off_line = np.median(tm_to_off_line) # Median team distance to opponent goal line
    
    sorted_def = np.sort(tm_to_def_line)
    st_closest_def_no_gk = sorted_def[1] if len(sorted_def) > 1 else sorted_def[0] # Deepest field player (covering defender)
    
    st_closest_top = np.min(tm_to_top) # Teammate closest to top sideline
    st_median_top = np.median(tm_to_top) # Median team width/distance to top
    st_closest_bottom = np.min(np.abs(teammates_pos[:, 1] - (-PITCH_Y_MAX))) # Teammate closest to bottom sideline
    
    return st_closest_off_line, st_closest_def_no_gk, st_closest_top, st_closest_bottom, st_median_off_line, st_median_top

def calc_s_is_gk(sender_pos, teammates_pos, defense_gate):
    """
    Sender Role Identifier (GK Check)
    
    Inputs: 
        - sender_pos: numpy array shape (2,)
        - teammates_pos: numpy array shape (10, 2)
        - defense_gate: numpy array shape (2,)
    Outputs: 
        - is_s_gk: int (1 if true, 0 if false)

    Identifies if the sender is effectively the goalkeeper (deepest player on team), as GK passing behavior differs significantly from outfield players.
    """
    s_to_def_gate = np.linalg.norm(sender_pos - defense_gate)
    tm_to_def_gate = np.linalg.norm(teammates_pos - defense_gate, axis=1)
    is_s_gk = 1 if s_to_def_gate == np.min(tm_to_def_gate) else 0 # 1 if sender is the deepest player
    return is_s_gk

# =============================================================================
# RECEIVER FEATURE HELPERS
# =============================================================================

def calc_r_zones(rec_pos, side):
    """
    Receiver Pitch Zone Classifier
    
    Inputs: 
        - rec_pos: numpy array shape (2,)
        - side: integer (1 or -1)
    Outputs: 
        - r_back, r_mid, r_front: 3 separate integers (0 or 1)

    Categorizes the potential receiver's position into defensive, middle, or attacking thirds to provide tactical context for the pass.
    """
    rec_x_norm = rec_pos[0] * side
    # Thresholds converted to CM (17.5m -> 1750cm)
    r_back = 1 if rec_x_norm < -1750 else 0 
    r_mid = 1 if -1750 <= rec_x_norm <= 1750 else 0 
    r_front = 1 if rec_x_norm > 1750 else 0 
    return r_back, r_mid, r_front

def calc_r_landmarks(rec_pos, offense_gate, defense_gate):
    """
    Receiver Landmark Distance Calculator
    
    Inputs: 
        - rec_pos: numpy array shape (2,)
        - offense_gate: numpy array shape (2,)
        - defense_gate: numpy array shape (2,)
    Outputs: 
        - r_to_off_gate: float
        - r_to_def_gate: float
        - r_to_center: float
        - is_r_center_circle: int (0 or 1)

    Measures the receiver's spatial relationship to goals and center circle to evaluate the strategic value of their position.
    """
    r_to_off_gate = np.linalg.norm(rec_pos - offense_gate) # Distance to target goal
    r_to_def_gate = np.linalg.norm(rec_pos - defense_gate) # Distance to own goal
    r_to_center = np.linalg.norm(rec_pos) # Distance to center spot
    # Center circle radius converted to CM (9.15m -> 915cm)
    is_r_center_circle = 1 if r_to_center < 915 else 0 
    return r_to_off_gate, r_to_def_gate, r_to_center, is_r_center_circle

def calc_r_geometry(rec_pos, sender_pos, side):
    """
    Pass Geometry Calculator
    
    Inputs: 
        - rec_pos: numpy array shape (2,)
        - sender_pos: numpy array shape (2,)
        - side: integer
    Outputs: 
        - norm_r_s_x_diff: float (Forward progress distance)
        - is_r_off_dir: int (1 if pass is forward)
        - abs_y_diff: float (Lateral pass width)

    Calculates the physical characteristics of the potential pass, distinguishing between forward progressive passes, back passes, and lateral switches.
    """
    norm_r_s_x_diff = (rec_pos[0] - sender_pos[0]) * side # Longitudinal distance (forward/backward)
    is_r_off_dir = 1 if norm_r_s_x_diff > 0 else 0 # 1 if pass is forward
    abs_y_diff = abs(rec_pos[1] - sender_pos[1]) # Lateral distance (width of pass)
    return norm_r_s_x_diff, is_r_off_dir, abs_y_diff

def calc_r_pressure(rec_pos, r_tms_pos, r_opps_pos):
    """
    Receiver Context/Pressure Calculator
    
    Inputs: 
        - rec_pos: numpy array shape (2,)
        - r_tms_pos: numpy array shape (N, 2) (Receiver's teammates)
        - r_opps_pos: numpy array shape (M, 2) (Receiver's opponents)
    Outputs: 
        - r_closest_tm: float
        - r_closest_3_tm: float
        - r_closest_opp: float
        - r_closest_3_opp: float

    Quantifies the density of players around the receiver to determine if they are marked (under pressure) or open (supported).
    """
    r_tm_dists = np.linalg.norm(r_tms_pos - rec_pos, axis=1)
    # Epsilon converted to CM (0.001m -> 0.1cm)
    r_tm_dists = r_tm_dists[r_tm_dists > 0.1] 
    
    r_opp_dists = np.linalg.norm(r_opps_pos - rec_pos, axis=1)
    
    r_closest_tm = np.min(r_tm_dists) if len(r_tm_dists) > 0 else 10000 # Distance to nearest support player
    r_closest_3_tm = np.mean(np.sort(r_tm_dists)[:3]) if len(r_tm_dists) >= 3 else r_closest_tm # Avg dist to 3 nearest supports
    
    r_closest_opp = np.min(r_opp_dists) if len(r_opp_dists) > 0 else 10000 # Distance to nearest pressure player
    r_closest_3_opp = np.mean(np.sort(r_opp_dists)[:3]) if len(r_opp_dists) >= 3 else r_closest_opp # Avg dist to 3 nearest pressure players
    
    return r_closest_tm, r_closest_3_tm, r_closest_opp, r_closest_3_opp

def calc_r_rel_sender(rec_pos, sender_pos, r_tms_pos, r_opps_pos):
    """
    Sender-Receiver Triangulation Calculator
    
    Inputs: 
        - rec_pos: numpy array shape (2,)
        - sender_pos: numpy array shape (2,)
        - r_tms_pos: numpy array shape (N, 2)
        - r_opps_pos: numpy array shape (M, 2)
    Outputs: 
        - r_closest_tm_to_sender: float
        - r_closest_opp_to_sender: float

    Measures the distance from the *sender* to the receiver's neighbors, helping evaluate if the pass is into a crowded zone relative to the sender's view.
    """
    # Neighbors relative to sender
    if len(r_tms_pos) > 0:
        full_tm_dists = np.linalg.norm(r_tms_pos - rec_pos, axis=1)
        full_tm_dists[full_tm_dists < 0.1] = 999999 # Updated epsilon
        closest_tm_idx = np.argmin(full_tm_dists)
        r_closest_tm_to_sender = np.linalg.norm(r_tms_pos[closest_tm_idx] - sender_pos) # Sender dist to receiver's nearest support
    else:
        r_closest_tm_to_sender = 10000

    if len(r_opps_pos) > 0:
        r_opp_dists = np.linalg.norm(r_opps_pos - rec_pos, axis=1)
        closest_opp_idx = np.argmin(r_opp_dists)
        r_closest_opp_to_sender = np.linalg.norm(r_opps_pos[closest_opp_idx] - sender_pos) # Sender dist to receiver's nearest pressure
    else:
        r_closest_opp_to_sender = 10000
        
    return r_closest_tm_to_sender, r_closest_opp_to_sender

def calc_r_is_gk(rec_pos, r_tms_pos, defense_gate, is_rec_same_team, offense_gate):
    """
    Receiver Role Identifier (GK Check)
    
    Inputs: 
        - rec_pos: numpy array shape (2,)
        - r_tms_pos: numpy array shape (N, 2)
        - defense_gate, offense_gate: numpy array shape (2,)
        - is_rec_same_team: int (0 or 1)
    Outputs: 
        - r_is_gk: int (1 if true, 0 if false)

    Identifies if the potential receiver is the goalkeeper, as backpasses to the GK are distinct safe-option plays.
    """
    own_goal = defense_gate if is_rec_same_team else offense_gate
    r_is_gk = 1 if np.linalg.norm(rec_pos - own_goal) == np.min(np.linalg.norm(r_tms_pos - own_goal, axis=1)) else 0
    return r_is_gk

def calc_r_ranks(rec_pos, sender_pos, r_tms_pos, r_opps_pos, offense_gate, defense_gate, is_rec_same_team):
    """
    Receiver Relative Ranking Calculator
    
    Inputs: 
        - rec_pos, sender_pos: numpy arrays shape (2,)
        - r_tms_pos, r_opps_pos: numpy arrays shape (N, 2)
        - offense_gate, defense_gate: numpy arrays shape (2,)
        - is_rec_same_team: int
    Outputs: 
        - 8 rank integers (Ranking receiver against all other players for various distance metrics)

    Comparatively ranks the receiver against all other players (e.g., 'Is this receiver the closest player to the goal?'), providing hierarchical context to the decision.
    """
    # Determine the target goal (Attacking gate)
    r_target_gate = offense_gate if is_rec_same_team else defense_gate
    
    # --- 1. Rank by Distance to Goal (How advanced are they?) ---
    r_tm_to_gate = np.linalg.norm(r_tms_pos - r_target_gate, axis=1)
    r_opp_to_gate = np.linalg.norm(r_opps_pos - r_target_gate, axis=1)
    r_dist_gate = np.linalg.norm(rec_pos - r_target_gate)
    
    # Rank 1 = Closest teammate to goal (Most advanced attacker)
    rank_r_gate_tm = np.sum(r_tm_to_gate < r_dist_gate) + 1 
    # Rank 1 = Closest player to goal compared to opponents (Breakaway potential)
    rank_r_gate_opp = np.sum(r_opp_to_gate < r_dist_gate) + 1 
    
    # --- 2. Rank by Width (Top Sideline) ---
    top_sideline_y = PITCH_Y_MAX
    r_dist_top = abs(rec_pos[1] - top_sideline_y)
    r_tm_to_top = np.abs(r_tms_pos[:, 1] - top_sideline_y)
    r_opp_to_top = np.abs(r_opps_pos[:, 1] - top_sideline_y)
    
    # Rank 1 = Widest teammate on the top flank
    rank_r_top_tm = np.sum(r_tm_to_top < r_dist_top) + 1 
    # Rank 1 = Wider than any opponent on that side (Unmarked on wing)
    rank_r_top_opp = np.sum(r_opp_to_top < r_dist_top) + 1 

    # --- 3. Rank by Width (Bottom Sideline) ---
    bottom_sideline_y = -PITCH_Y_MAX
    r_dist_bottom = abs(rec_pos[1] - bottom_sideline_y)
    r_tm_to_bottom = np.abs(r_tms_pos[:, 1] - bottom_sideline_y)
    r_opp_to_bottom = np.abs(r_opps_pos[:, 1] - bottom_sideline_y)
    
    # Rank 1 = Widest teammate on the bottom flank
    rank_r_bottom_tm = np.sum(r_tm_to_bottom < r_dist_bottom) + 1
    # Rank 1 = Wider than any opponent on the bottom flank
    rank_r_bottom_opp = np.sum(r_opp_to_bottom < r_dist_bottom) + 1
    
    # --- 4. Rank by Proximity to Sender (Support options) ---
    dist_s_r = np.linalg.norm(sender_pos - rec_pos)
    r_tm_to_s = np.linalg.norm(r_tms_pos - sender_pos, axis=1)
    r_opp_to_s = np.linalg.norm(r_opps_pos - sender_pos, axis=1)
    
    # Rank 1 = Closest support option for the ball carrier (Safety pass)
    rank_r_s_tm = np.sum(r_tm_to_s < dist_s_r) + 1 
    # Rank 1 = Closer to sender than any opponent (Pressing context)
    rank_r_s_opp = np.sum(r_opp_to_s < dist_s_r) + 1 
    
    return rank_r_gate_tm, rank_r_gate_opp, rank_r_top_tm, rank_r_top_opp, rank_r_bottom_tm, rank_r_bottom_opp, rank_r_s_tm, rank_r_s_opp

# =============================================================================
# FEATURE EXTRACTION CONTEXT & MAIN
# =============================================================================

def parse_match_context(row, side):
    """
    Match Snapshot Parser
    
    Inputs: 
        - row: pandas Series (Single row of raw match data)
        - side: integer (1 or -1)
    Outputs: 
        - ctx: dictionary (Contains structured numpy arrays for positions and metadata)

    Extracts, normalizes, and organizes raw coordinate data into a structured context object containing teammates, opponents, and game geometry.
    """
    sid = int(row.sender_id)
    sender_pos = np.array([row[f"x_{sid}"], row[f"y_{sid}"]])
    
    is_team1 = (sid <= 11)
    teammate_ids = range(1, 12) if is_team1 else range(12, 23)
    opponent_ids = range(12, 23) if is_team1 else range(1, 12)
    
    teammates_pos = np.array([[row[f"x_{k}"], row[f"y_{k}"]] for k in teammate_ids])
    opponents_pos = np.array([[row[f"x_{k}"], row[f"y_{k}"]] for k in opponent_ids])
    
    # Gates: Offense is target, Defense is own goal
    offense_gate = np.array([GOAL_X * side, 0])
    defense_gate = np.array([-GOAL_X * side, 0])
    
    return {
        "sid": sid,
        "sender_pos": sender_pos,
        "is_team1": is_team1,
        "teammates_pos": teammates_pos,
        "opponents_pos": opponents_pos,
        "offense_gate": offense_gate,
        "defense_gate": defense_gate,
        "side": side
    }

def get_sender_features(ctx):
    """
    Sender Feature Aggregator
    
    Inputs: 
        - ctx: dictionary (Match context)
    Outputs: 
        - features: list (Sender-specific feature values)
        - meta: dictionary (Metadata needed for subsequent receiver calculations)

    Orchestrates the calculation of all metrics specific to the ball carrier, which remain constant across all potential pass options for a given snapshot.
    """
    sender_pos = ctx["sender_pos"]
    teammates_pos = ctx["teammates_pos"]
    opponents_pos = ctx["opponents_pos"]
    offense_gate = ctx["offense_gate"]
    defense_gate = ctx["defense_gate"]
    side = ctx["side"]

    # 1. Zones
    s_back, s_mid, s_front = calc_s_zones(sender_pos, side)
    
    # 2. Distances to landmarks
    s_to_off_gate, s_to_def_gate, s_to_center = calc_s_landmark_dists(sender_pos, offense_gate, defense_gate)
    
    # 3. Distances to players
    s_closest_tm, s_closest_3_tm, s_closest_opp, s_closest_3_opp = calc_s_player_dists(sender_pos, teammates_pos, opponents_pos)
    
    # 4. Ranks
    rank_s_off_gate_tm, rank_s_off_gate_opp, rank_s_top_tm, rank_s_top_opp = calc_s_ranks(sender_pos, teammates_pos, opponents_pos, offense_gate)

    # 5. Team Geometry
    st_closest_off_line, st_closest_def_no_gk, st_closest_top, st_closest_bottom, st_median_off_line, st_median_top = calc_team_geometry(teammates_pos, offense_gate, defense_gate)
    
    # 6. Is GK
    is_s_gk = calc_s_is_gk(sender_pos, teammates_pos, defense_gate)
    
    # Package features
    features = [
        s_back, s_mid, s_front,
        s_to_off_gate, s_to_def_gate, s_to_center,
        is_s_gk,
        s_closest_tm, s_closest_3_tm,
        s_closest_opp, s_closest_3_opp,
        rank_s_off_gate_tm, rank_s_off_gate_opp,
        rank_s_top_tm, rank_s_top_opp,
        st_closest_off_line, st_closest_def_no_gk,
        st_closest_top, st_closest_bottom,
        st_median_off_line, st_median_top
    ]
    
    # Return features + context for receiver features (like sender zone)
    meta = {
        "s_back": s_back, "s_mid": s_mid, "s_front": s_front,
        "s_closest_tm_val": s_closest_tm # Used for logic checks if needed
    }
    return features, meta

def get_receiver_features(pid, row, ctx, sender_meta):
    """
    Receiver Feature Aggregator
    
    Inputs: 
        - pid: int (Receiver Player ID)
        - row: pandas Series
        - ctx: dictionary
        - sender_meta: dictionary
    Outputs: 
        - features: list (Receiver-specific feature values)
        - rec_pos: numpy array shape (2,)

    Calculates the full suite of spatial, relative, and contextual features for a single candidate receiver.
    """
    sender_pos = ctx["sender_pos"]
    is_team1 = ctx["is_team1"]
    offense_gate = ctx["offense_gate"]
    defense_gate = ctx["defense_gate"]
    side = ctx["side"]
    teammates_pos = ctx["teammates_pos"]
    opponents_pos = ctx["opponents_pos"]
    
    rec_pos = np.array([row[f"x_{pid}"], row[f"y_{pid}"]])
    
    # Basic Info
    is_rec_same_team = 1 if (is_team1 and pid <= 11) or (not is_team1 and pid > 11) else 0 # 1 if receiver is teammate of sender
    dist_s_r = np.linalg.norm(sender_pos - rec_pos) # Distance between sender and receiver
    
    # 1. Zones
    r_back, r_mid, r_front = calc_r_zones(rec_pos, side)
    
    # Check if in same field zone
    s_r_same_field = 1 if (sender_meta["s_back"]==r_back and sender_meta["s_mid"]==r_mid and sender_meta["s_front"]==r_front) else 0 # 1 if sender and receiver in same third
    
    # 2. Landmarks
    r_to_off_gate, r_to_def_gate, r_to_center, is_r_center_circle = calc_r_landmarks(rec_pos, offense_gate, defense_gate)
    
    # 3. Relative Geometry
    norm_r_s_x_diff, is_r_off_dir, abs_y_diff = calc_r_geometry(rec_pos, sender_pos, side)
    
    # Context Swapping (Who are the receiver's teammates?)
    if is_rec_same_team:
        r_tms_pos = teammates_pos
        r_opps_pos = opponents_pos
    else:
        r_tms_pos = opponents_pos
        r_opps_pos = teammates_pos
        
    # 4. Pressure / Support
    r_closest_tm, r_closest_3_tm, r_closest_opp, r_closest_3_opp = calc_r_pressure(rec_pos, r_tms_pos, r_opps_pos)
    
    # 5. Neighbors relative to Sender
    r_closest_tm_to_sender, r_closest_opp_to_sender = calc_r_rel_sender(rec_pos, sender_pos, r_tms_pos, r_opps_pos)
        
    # 6. Is GK
    r_is_gk = calc_r_is_gk(rec_pos, r_tms_pos, defense_gate, is_rec_same_team, offense_gate)

    # 7. Ranks
    rank_r_gate_tm, rank_r_gate_opp, rank_r_top_tm, rank_r_top_opp, rank_r_bottom_tm, rank_r_bottom_opp, rank_r_s_tm, rank_r_s_opp = calc_r_ranks(
        rec_pos, sender_pos, r_tms_pos, r_opps_pos, offense_gate, defense_gate, is_rec_same_team
    )

    return [
        dist_s_r, is_rec_same_team,
        r_back, r_mid, r_front, s_r_same_field,
        r_to_off_gate, r_to_def_gate,
        norm_r_s_x_diff, is_r_off_dir, abs_y_diff,
        r_to_center, is_r_center_circle, r_is_gk,
        r_closest_tm, r_closest_3_tm,
        r_closest_opp, r_closest_3_opp,
        r_closest_tm_to_sender, r_closest_opp_to_sender,
        rank_r_gate_tm, rank_r_gate_opp,
        rank_r_top_tm, rank_r_top_opp,
        rank_r_bottom_tm, rank_r_bottom_opp,
        rank_r_s_tm, rank_r_s_opp
    ], rec_pos

def get_passing_path_features(sender_pos, rec_pos, opponents_pos, is_rec_same_team):
    """
    Passing Lane Safety Evaluator
    
    Inputs: 
        - sender_pos, rec_pos: numpy arrays shape (2,)
        - opponents_pos: numpy array shape (11, 2)
        - is_rec_same_team: int
    Outputs: 
        - path_features: list of 5 floats/ints (Distance to line, blocker count, min angles)

    Evaluates the "cleanness" of the passing lane by calculating opponent proximity to the direct line and the number of intervening defenders.
    """
    dist_s_r = np.linalg.norm(sender_pos - rec_pos)
    dists_to_line = []
    dangerous_count = 0
    pass_angles = []
    
    vec_s_r = rec_pos - sender_pos
    
    for opp in opponents_pos:
        d_line = dist_point_to_segment(opp, sender_pos, rec_pos)
        dists_to_line.append(d_line)
        
        d_o_s = np.linalg.norm(opp - sender_pos)
        d_o_r = np.linalg.norm(opp - rec_pos)
        if d_o_s < dist_s_r and d_o_r < dist_s_r:
            dangerous_count += 1
            vec_s_o = opp - sender_pos
            angle = get_angle_between_vectors(vec_s_r, vec_s_o)
            pass_angles.append(angle)
    
    dists_to_line.sort()
    min_opp_dist_line = dists_to_line[0] if len(dists_to_line) > 0 else 100
    sec_opp_dist_line = dists_to_line[1] if len(dists_to_line) > 1 else 100
    thd_opp_dist_line = dists_to_line[2] if len(dists_to_line) > 2 else 100
    
    if not is_rec_same_team:
        min_opp_dist_line = -1
        
    min_pass_angle = np.min(pass_angles) if len(pass_angles) > 0 else np.pi
    
    return [
        min_opp_dist_line, sec_opp_dist_line, thd_opp_dist_line,
        dangerous_count, min_pass_angle
    ]

# =============================================================================
# FEATURE EXTRACTION CORE
# =============================================================================

def calculate_row_features(row, side):
    """
    Single-Row Feature Processor
    
    Inputs: 
        - row: pandas Series (Single raw match snapshot)
        - side: integer (Playing direction)
    Outputs: 
        - results: list of 22 lists (Feature vectors for all 22 players in this snapshot)

    Orchestrates the entire feature extraction pipeline for a single moment in the match, generating 22 samples (one per player) from one input row.
    """
    # 1. Parse Context
    ctx = parse_match_context(row, side)
    
    # 2. Sender Features (Game State removed)
    sender_feats, sender_meta = get_sender_features(ctx)
    
    # Combine
    common_feats = sender_feats
    
    results = []
    
    # 3. Receiver Loop
    for pid in range(1, 23):
        # Extract receiver spatial/relational features
        rec_feats, rec_pos = get_receiver_features(pid, row, ctx, sender_meta)
        
        # Extract passing path features (requires both S and R coords)
        path_feats = get_passing_path_features(ctx["sender_pos"], rec_pos, ctx["opponents_pos"], rec_feats[1]) # rec_feats[1] is is_rec_same_team
        
        # Combine: Common + Rec + Path
        # We need to insert path feats into the correct order to match original implementation
        # Original Order in Rec block:
        # [Dist, SameTeam, Zones(3), SameField, GateDists(2), RelGeom(3), Center(2), GK, 
        #  ClosestNeighbors(4), NeighborsToSender(2), 
        #  PATH_FEATS(5), 
        #  Ranks(8)] <-- NOW 8 Ranks (added bottom sideline)
        
        # rec_feats returned currently:
        # [Dist, SameTeam, Zones(3), SameField, GateDists(2), RelGeom(3), Center(2), GK, 
        #  ClosestNeighbors(4), NeighborsToSender(2), 
        #  Ranks(8)]
        
        # We splice path_feats before Ranks (last 8 elements)
        full_rec_feats = rec_feats[:-8] + path_feats + rec_feats[-8:]
        
        results.append(common_feats + full_rec_feats)
        
    return results

def make_dataset(X, y=None, sides=None):
    """
    Full Dataset Constructor
    
    Inputs: 
        - X: pandas DataFrame (Raw match data)
        - y: pandas DataFrame or None (Target labels)
        - sides: numpy array (Direction for each match)
    Outputs: 
        - X_df: pandas DataFrame (N_samples * 22, N_features)
        - y_df: pandas DataFrame (N_samples * 22, 1)

    Converts the entire raw input dataset (N events) into the structured (N*22 samples) format required for training the Random Forest/LGBM models.
    """
    data_list = []
    target_list = []
    
    print("Generating Advanced Features...")
    
    for i in range(len(X)):
        row = X.iloc[i]
        side = sides[i * 22]
        
        feats_matrix = calculate_row_features(row, side)
        data_list.extend(feats_matrix)
        
        if y is not None:
            true_receiver = y.iloc[i, 0]
            for pid in range(1, 23):
                target_list.append(1 if pid == true_receiver else 0)
    
    # Ensure these column names match extraction order exactly
    cols = [
        "s_back", "s_mid", "s_front",
        "s_to_off_gate", "s_to_def_gate", "s_to_center", "is_s_gk",
        "s_closest_tm", "s_closest_3_tm", "s_closest_opp", "s_closest_3_opp",
        "rank_s_off_gate_tm", "rank_s_off_gate_opp", "rank_s_top_tm", "rank_s_top_opp",
        "st_closest_off_line", "st_closest_def_no_gk", "st_closest_top", "st_closest_bottom",
        "st_median_off_line", "st_median_top",
        "distance", "is_same_team",
        "r_back", "r_mid", "r_front", "s_r_same_field",
        "r_to_off_gate", "r_to_def_gate",
        "norm_r_s_x_diff", "is_r_off_dir", "abs_y_diff",
        "r_to_center", "is_r_center_circle", "r_is_gk",
        "r_closest_tm", "r_closest_3_tm", "r_closest_opp", "r_closest_3_opp",
        "r_closest_tm_to_sender", "r_closest_opp_to_sender",
        "min_opp_dist_line", "sec_opp_dist_line", "thd_opp_dist_line",
        "num_dangerous_opps", "min_pass_angle",
        "rank_r_gate_tm", "rank_r_gate_opp",
        "rank_r_top_tm", "rank_r_top_opp",
        "rank_r_bottom_tm", "rank_r_bottom_opp",
        "rank_r_s_tm", "rank_r_s_opp"
    ]
    
    X_df = pd.DataFrame(data_list, columns=cols)
    y_df = pd.DataFrame(target_list, columns=["pass"]) if target_list else None
    
    return X_df, y_df

# =============================================================================
# DATA LOADING & SIDES
# =============================================================================

def load_data(path):
    """
    Data Loader
    
    Inputs: 
        - path: string (File path to CSV)
    Outputs: 
        - df: pandas DataFrame (Loaded data)

    Loads the raw CSV data. Input data is assumed to be in centimeters, matching the global configuration.
    """
    df = pd.read_csv(path, index_col=0)
    return df

def augment_data(X, y=None):
    """
    Dataset Augmentor (Mirroring)
    
    Inputs: 
        - X: pandas DataFrame (Input features)
        - y: pandas DataFrame or None (Targets)
    Outputs: 
        - X_aug, y_aug: DataFrames (Doubled size)

    Doubles the effective dataset size by mirroring every play across the Y-axis (left/right flip), helping the model learn symmetric spatial features.
    """
    print("Augmenting data (flipping Y)...")
    X_flipped = X.copy()
    # Flip all y columns (y_1 to y_22)
    y_cols = [c for c in X.columns if c.startswith('y_')]
    X_flipped[y_cols] = -X_flipped[y_cols]
    
    # Reset index for clean concat, but we must track logic manually anyway
    X_aug = pd.concat([X, X_flipped], ignore_index=True)
    
    y_aug = None
    if y is not None:
        y_aug = pd.concat([y, y], ignore_index=True)
        
    return X_aug, y_aug

def detect_sides(df):
    """
    Attacking Direction Detector
    
    Inputs: 
        - df: pandas DataFrame (Raw coordinate data)
    Outputs: 
        - sides: numpy array shape (N_samples * 22,) (1 or -1)

    Automatically infers the attacking direction (left-to-right or right-to-left) for each match snapshot based on team positioning.
    """
    # We use the median position of Team 1 players to determine their half.
    # This is more robust than min/max which can be skewed by a single player.
    t1_x_cols = [f'x_{i}' for i in range(1, 12)]
    
    # Calculate median X for Team 1 in every sample
    t1_medians = df[t1_x_cols].median(axis=1).values
    
    # If Team 1 median is negative, they are on the left, so they attack Right (+1)
    # If Team 1 median is positive, they are on the right, so they attack Left (-1)
    game_sides = np.where(t1_medians < 0, 1, -1)
    
    # Expand to match the shape required by make_dataset (one side value per player row)
    sides = np.repeat(game_sides, 22)
    return sides

# =============================================================================
# MODEL CLASS
# =============================================================================

class BaggedEnsemble(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_models=5):
        """
        Ensemble Initialization
        
        Inputs: 
            - n_models: int (Number of base stacking models to train)
        Outputs: 
            - None

        Configures the bagging ensemble, defining how many independent stacking predictors will be trained and averaged.
        """
        self.n_models = n_models
        self.models = []
        self.feature_importances_ = None
        self.feature_names_ = None
        self.val_scores_history_ = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Ensemble Training Loop
        
        Inputs: 
            - X, y: Training data and labels
            - X_val, y_val: Validation data and labels (optional)
        Outputs: 
            - self: Fitted estimator

        Trains multiple independent StackingClassifiers (LGBM+RF+ExtraTrees -> LogReg) using different random seeds for robustness.
        """
        self.models = []
        self.val_scores_history_ = []
        print(f"Training Bagged Ensemble ({self.n_models} generations)...")
        
        SAFE_N_JOBS = 4

        # REMOVED CLASS WEIGHTS to ensure calibrated probabilities for Brier Score

        for i in range(self.n_models):
            seed = 42 + i
            
            lgbm = LGBMClassifier(
                n_estimators=220,
                max_depth=8, 
                num_leaves=32,
                learning_rate=0.05,
                objective='binary',
                random_state=seed,
                n_jobs=SAFE_N_JOBS,
                verbose=-1
            )
            
            rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=8,
                random_state=seed,
                n_jobs=SAFE_N_JOBS 
            )

            et = ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=8,
                random_state=seed,
                n_jobs=SAFE_N_JOBS 
            )
            
            base_stacking = StackingClassifier(
                estimators=[('lgbm', lgbm), ('rf', rf), ('et', et)], 
                final_estimator=LogisticRegression(),
                cv=3,
                n_jobs=SAFE_N_JOBS 
            )
            
            base_stacking.fit(X, y.values.ravel())
            self.models.append(base_stacking)
            
            if X_val is not None and y_val is not None:
                acc = self.score_multiclass(X_val, y_val)
                self.val_scores_history_.append(acc)
        
        self._calculate_feature_importances(X.columns)
        return self

    def _calculate_feature_importances(self, features):
        """
        Feature Importance Aggregator
        
        Inputs: 
            - features: list of strings (Column names)
        Outputs: 
            - None (Sets internal self.feature_importances_ attribute)

        Averages the feature importance scores from all underlying tree models to determine which metrics drive the ensemble's decisions.
        """
        feature_importances = np.zeros(len(features))
        counts = 0
        for stack in self.models:
            for est in stack.estimators_:
                if hasattr(est, 'feature_importances_'):
                    fi = est.feature_importances_
                    if fi.sum() > 0: fi = fi / fi.sum()
                    feature_importances += fi
                    counts += 1
        if counts > 0:
            self.feature_importances_ = feature_importances / counts
            self.feature_names_ = features

    def predict_proba(self, X):
        """
        Ensemble Probability Predictor
        
        Inputs: 
            - X: pandas DataFrame (Features to predict)
        Outputs: 
            - probs: numpy array (Averaged class probabilities)

        Generates stable predictions by averaging the probability outputs of all trained models in the bagging ensemble.
        """
        probs = sum(m.predict_proba(X) for m in self.models) / len(self.models)
        return probs

    def score_multiclass(self, X, y):
        """
        Multiclass Accuracy Scorer
        
        Inputs: 
            - X, y: Test data and labels
        Outputs: 
            - accuracy: float

        Evaluates the ensemble's performance by converting binary probabilities back into a multiclass winner (1 out of 22) and checking accuracy.
        """
        probs = self.predict_proba(X)[:, 1]
        probs_2d = probs.reshape(-1, 22)
        
        # Apply Softmax for scoring to match pipeline
        probs_2d = softmax(probs_2d)
        
        y_2d = y.values.ravel().reshape(-1, 22)
        preds = np.argmax(probs_2d, axis=1)
        trues = np.argmax(y_2d, axis=1)
        return np.mean(preds == trues)

# =============================================================================
# GRAPHS & OUTPUT
# =============================================================================

def ensure_graph_dir():
    """
    Directory Manager
    
    Inputs: 
        - None
    Outputs: 
        - None (Side effect: Creates folder)

    Checks for the existence of the graph output directory and creates it if missing to prevent file I/O errors.
    """
    if not os.path.exists(GRAPH_OUTPUT_DIR): os.makedirs(GRAPH_OUTPUT_DIR)

def plot_feature_importance(model, filename="feature_importance.png"):
    """
    Feature Importance Plotter
    
    Inputs: 
        - model: BaggedEnsemble instance
        - filename: string
    Outputs: 
        - None (Saves .png file)

    Generates and saves a bar chart visualizing the top 30 most predictive features according to the model ensemble.
    """
    if model.feature_importances_ is None: return
    ensure_graph_dir()
    df_imp = pd.DataFrame({'feature': model.feature_names_, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=df_imp.head(30), hue='feature', palette='viridis', legend=False)
    plt.title('Top 30 Advanced Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_OUTPUT_DIR, filename))
    plt.close()

def plot_prediction_confidence(probas, filename="confidence_distribution.png"):
    """
    Confidence Distribution Plotter
    
    Inputs: 
        - probas: numpy array shape (N_samples, 22)
        - filename: string
    Outputs: 
        - None (Saves .png file)

    Visualizes the histogram of the model's maximum confidence scores to diagnose if the model is overconfident or underconfident.
    """
    ensure_graph_dir()
    max_probs = np.max(probas, axis=1)
    plt.figure(figsize=(10, 6))
    sns.histplot(max_probs, bins=50, kde=True, color='purple')
    plt.title('Prediction Confidence Distribution (After Softmax)')
    plt.savefig(os.path.join(GRAPH_OUTPUT_DIR, filename))
    plt.close()

def plot_learning_curve(scores, filename="learning_curve.png"):
    """
    Learning Curve Plotter
    
    Inputs: 
        - scores: list of floats (Validation scores per fold)
        - filename: string
    Outputs: 
        - None (Saves .png file)

    Plots the validation accuracy across cross-validation folds to visualize model stability and performance variance.
    """
    if not scores: return
    ensure_graph_dir()
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores) + 1), scores, marker='o', color='b')
    plt.title('Bagging Validation Accuracy')
    plt.savefig(os.path.join(GRAPH_OUTPUT_DIR, filename))
    plt.close()

def plot_per_player_precision(precisions_list, filename="player_precision_boxplot.png"):
    """
    Player Precision Boxplotter
    
    Inputs:
        - precisions_list: list of numpy arrays (Each array contains 22 precision scores for a fold)
        - filename: string
    Outputs:
        - None (Saves .png file)
        
    Visualizes the stability of the model's precision for each specific player ID across all cross-validation folds.
    """
    ensure_graph_dir()
    # Convert list of arrays to DataFrame
    # Rows: Folds, Cols: Players
    df = pd.DataFrame(precisions_list, columns=[f'P{i}' for i in range(1, 23)])
    
    # Melt for plotting
    df_melted = df.melt(var_name='Player', value_name='Precision')
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Player', y='Precision', data=df_melted, palette='coolwarm')
    plt.title('Precision Distribution per Player across 10 Folds')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_OUTPUT_DIR, filename))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png"):
    """
    Confusion Matrix Heatmap Plotter
    
    Inputs:
        - y_true: list/array (True class indices)
        - y_pred: list/array (Predicted class indices)
        - filename: string
    Outputs:
        - None (Saves .png file)
        
    Generates a heatmap showing the frequency of correct and incorrect predictions between all 22 player classes.
    """
    ensure_graph_dir()
    cm = confusion_matrix(y_true, y_pred, labels=range(22))
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'P{i}' for i in range(1, 23)],
                yticklabels=[f'P{i}' for i in range(1, 23)])
    plt.xlabel('Predicted Player')
    plt.ylabel('True Receiver')
    plt.title('Aggregated Confusion Matrix (All Folds)')
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_OUTPUT_DIR, filename))
    plt.close()

def write_submission(predictions=None, probas=None, estimated_score=0, file_name="submission", date=True, indexes=None):
    """
    Kaggle Submission Writer
    
    Inputs: 
        - probas: numpy array shape (N, 22) (Class probabilities)
        - estimated_score: float (Local CV score for metadata)
        - file_name: string (Base name)
        - indexes: numpy array (Row IDs)
    Outputs: 
        - file_name: string (Path to saved CSV)

    Formats the final model predictions and probabilities into the specific CSV structure required for competition submission.
    """
    if date: file_name = '{}_{}'.format(file_name, time.strftime('%d-%m-%Y_%Hh%M'))
    file_name = '{}.csv'.format(file_name)
    n_samples = len(probas) if probas is not None else len(predictions)
    if indexes is None: indexes = np.arange(n_samples)
    
    # If we have probabilities, derive predictions from them
    if predictions is None:
        predictions = np.zeros((n_samples, ))
        for i in range(n_samples):
            # argmax returns index 0-21, we need 1-22
            predictions[i] = np.argmax(probas[i]) + 1

    with open(file_name, 'w') as handle:
        header = '"Id","Predicted",' + ','.join([f'"P_{j}"' for j in range(1,23)])
        handle.write(header+"\n")
        first_line = '"Estimation",{},'.format(estimated_score) + ','.join(['0']*22)
        handle.write(first_line+"\n")
        for i in range(n_samples):
            line = "{},{:0.0f},".format(indexes[i], predictions[i]) + ','.join(map(str, probas[i, :]))
            handle.write(line+"\n")
    return file_name

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(input_file, output_file):
    """
    End-to-End Pipeline Executor
    
    Inputs: 
        - input_file: string (Path to training data)
        - output_file: string (Path to training labels)
    Outputs: 
        - val_score: float (Average CV accuracy)
        - models_list: list (Trained BaggedEnsemble instances)

    Controls the complete workflow: loading data, augmenting it, running 10-fold cross-validation, training ensembles, and reporting performance metrics.
    """
    prefix = os.path.dirname(os.path.abspath(__file__)) + '/'
    # load_data now handles unit conversion
    X_original = load_data(prefix + input_file)
    y_original = load_data(prefix + output_file)
    
    # AUGMENTATION: Double the dataset by flipping Y
    X_full, y_full = augment_data(X_original, y_original)
    
    s_full = detect_sides(X_full)

    # Pre-generate features for ALL data (2x size)
    # The splitting will happen on INDICES to ensure no overlap/leakage
    X_full_f, y_full_f = make_dataset(X_full, y_full, s_full)
    
    val_score = 0
    n_original_passes = len(X_original)
    
    # 10-Fold Cross-Validation becomes the main training loop
    # We collect all models trained on folds to serve as the final ensemble
    print("\n--- Training on 10-Fold Augmented Dataset (Stacking Models) ---")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    
    # We split based on ORIGINAL passes (0 to N-1).
    # We then include the corresponding mirrored pass (i + N) in the same fold.
    # This ensures 0 overlap between train/val groups.
    
    fold_scores = []
    final_models_list = []
    
    # Containers for report graphs
    fold_precisions = []
    all_val_trues = []
    all_val_preds = []
    
    pass_indices = np.arange(n_original_passes)
    
    for fold_idx, (train_idx_passes, val_idx_passes) in enumerate(kf.split(pass_indices)):
        print(f"Fold {fold_idx+1}/10")
        
        # 1. Expand indices to include mirrored versions
        # If pass 'i' is in train, then 'i + n_original' (flipped version) is also in train
        train_idx_extended = np.concatenate([train_idx_passes, train_idx_passes + n_original_passes])
        val_idx_extended = np.concatenate([val_idx_passes, val_idx_passes + n_original_passes])
        
        # 2. Map Pass Indices -> Augmented Row Indices (x22)
        # vectorized generation of 22 rows per pass
        train_idx_final = (train_idx_extended[:, None] * 22 + np.arange(22)).flatten()
        val_idx_final = (val_idx_extended[:, None] * 22 + np.arange(22)).flatten()
        
        X_train_f = X_full_f.iloc[train_idx_final]
        y_train_f = y_full_f.iloc[train_idx_final]
        X_val_f = X_full_f.iloc[val_idx_final]
        y_val_f = y_full_f.iloc[val_idx_final]
        
        # Use a smaller ensemble inside each fold (e.g., 3 generations)
        # Since we have 10 folds, the total final ensemble will have 30 stacking models!
        model = BaggedEnsemble(n_models=3) 
        model.fit(X_train_f, y_train_f, X_val=X_val_f, y_val=y_val_f)
        final_models_list.append(model)
        
        score = model.score_multiclass(X_val_f, y_val_f)
        fold_scores.append(score)
        print(f"  > Fold {fold_idx+1} Val Score: {score:.4f}")
        
        # --- NEW: Precision per Class Calculation ---
        # Get raw probabilities (binary for "is receiver")
        raw_probs = model.predict_proba(X_val_f)[:, 1]
        
        # Reshape to (n_samples, 22 candidates)
        probs_2d = raw_probs.reshape(-1, 22)
        probs_2d = softmax(probs_2d) # Apply softmax across the 22 candidates
        
        # Get ground truth
        y_val_reshaped = y_val_f.values.ravel().reshape(-1, 22)
        
        # Convert to indices (0-21)
        val_preds_indices = np.argmax(probs_2d, axis=1)
        val_true_indices = np.argmax(y_val_reshaped, axis=1)
        
        # Calculate summary stats
        macro_prec = precision_score(val_true_indices, val_preds_indices, average='macro', zero_division=0)
        weighted_prec = precision_score(val_true_indices, val_preds_indices, average='weighted', zero_division=0)
        
        print(f"  > Macro Precision:    {macro_prec:.4f}")
        print(f"  > Weighted Precision: {weighted_prec:.4f}")
        
        # Calculate precision per class
        # labels=range(22) ensures we track all 22 possible receivers even if one isn't picked in this fold
        per_class_prec = precision_score(val_true_indices, val_preds_indices, average=None, labels=range(22), zero_division=0)
        
        # Accumulate for graphs
        fold_precisions.append(per_class_prec)
        all_val_trues.extend(val_true_indices)
        all_val_preds.extend(val_preds_indices)
        
        print("\n    --- Detailed Per-Player Precision (Fold {}) ---".format(fold_idx+1))
        for p_idx, p_score in enumerate(per_class_prec):
            print(f"      Player {p_idx+1}: {p_score:.4f}")
        print("-" * 50)
        # ---------------------------------------------
        
    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    val_score = avg_score
    print(f"\n>>> 10-Fold CV (Augmented) Mean Acc = {avg_score:.4f} (+/- {std_score:.4f})")
    
    print("\nGenerating Report Graphs...")
    plot_learning_curve(fold_scores, "cv_10fold_scores.png")
    plot_per_player_precision(fold_precisions)
    plot_confusion_matrix(all_val_trues, all_val_preds)
    
    # We return the list of ALL models trained across the folds.
    # The final prediction will be the average of these 10 models.
    return val_score, final_models_list

if __name__ == '__main__':
    # Adjust filenames if they differ
    score, models_list = run_pipeline('input_train_set.csv', 'output_train_set.csv')
    
    prefix = os.path.dirname(os.path.abspath(__file__)) + '/'
    try:
        X_test = load_data(prefix + 'input_test_set.csv')
        s_test = detect_sides(X_test)
        X_test_f, _ = make_dataset(X_test, None, s_test)
        
        print(f"\nPredicting with Ensemble of {len(models_list)} Fold Models...")
        
        # Accumulate probabilities from all fold models
        accumulated_probs = None
        
        for i, model in enumerate(models_list):
            print(f"  > Predicting with model from Fold {i+1}...")
            probs = model.predict_proba(X_test_f)[:, 1]
            if accumulated_probs is None:
                accumulated_probs = probs
            else:
                accumulated_probs += probs
        
        # Average the probabilities
        avg_probs = accumulated_probs / len(models_list)
        
        # Reshape to (N_samples, 22_players)
        probas_2d = avg_probs.reshape(-1, 22)
        
        # CRITICAL: Apply Softmax normalization for Brier Score optimization
        # This ensures the sum of probabilities for the 22 candidates is 1.0
        probas_final = softmax(probas_2d)
        
        plot_prediction_confidence(probas_final, "test_confidence_distribution_adv.png")
        
        results_dir = os.path.join(prefix, 'Results')
        if not os.path.exists(results_dir): os.makedirs(results_dir)
        fname = write_submission(probas=probas_final, estimated_score=score, file_name=os.path.join(results_dir, "advanced_features"), indexes=X_test.index)
        print(f"Submission file \"{fname}\" successfully written")
    except FileNotFoundError:
         print("Test set not found. Ensure input_test_set.csv is in the directory.")