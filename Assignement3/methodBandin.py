import os
import time
import datetime
from contextlib import contextmanager
from math import copysign as sign, acos, sqrt, pi, degrees

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from lightgbm import LGBMClassifier 
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import precision_score, confusion_matrix, accuracy_score

# =============================================================================
# CONFIGURATION
# =============================================================================

# Construct absolute path to "Graphs" folder relative to this script file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPH_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "Graphs")

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
    """
    start = time.time()
    yield
    elapsed = datetime.timedelta(seconds=time.time() - start)
    print(f'Duration of [{label}]: {elapsed}')

def dist_point_to_segment(p, a, b):
    """
    Geometric Distance Calculator
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
    """
    sender_x_norm = sender_pos[0] * side
    s_back = 1 if sender_x_norm < -1750 else 0 
    s_mid = 1 if -1750 <= sender_x_norm <= 1750 else 0 
    s_front = 1 if sender_x_norm > 1750 else 0 
    return s_back, s_mid, s_front

def calc_s_landmark_dists(sender_pos, offense_gate, defense_gate):
    """
    Sender Landmark Distance Calculator
    """
    s_to_off_gate = np.linalg.norm(sender_pos - offense_gate) 
    s_to_def_gate = np.linalg.norm(sender_pos - defense_gate) 
    s_to_center = np.linalg.norm(sender_pos) 
    return s_to_off_gate, s_to_def_gate, s_to_center

def calc_s_player_dists(sender_pos, teammates_pos, opponents_pos):
    """
    Sender Immediate Context Calculator
    """
    tm_dists = np.linalg.norm(teammates_pos - sender_pos, axis=1)
    tm_dists_excl_sender = tm_dists[tm_dists > 0.1] 
    opp_dists = np.linalg.norm(opponents_pos - sender_pos, axis=1)
    
    s_closest_tm = np.min(tm_dists_excl_sender) if len(tm_dists_excl_sender) > 0 else 10000 
    s_closest_3_tm = np.mean(np.sort(tm_dists_excl_sender)[:3]) if len(tm_dists_excl_sender) >= 3 else s_closest_tm 
    
    s_closest_opp = np.min(opp_dists) if len(opp_dists) > 0 else 10000 
    s_closest_3_opp = np.mean(np.sort(opp_dists)[:3]) if len(opp_dists) >= 3 else s_closest_opp 
    
    return s_closest_tm, s_closest_3_tm, s_closest_opp, s_closest_3_opp

def calc_s_ranks(sender_pos, teammates_pos, opponents_pos, offense_gate):
    """
    Sender Relative Ranking Calculator
    """
    s_to_off_gate = np.linalg.norm(sender_pos - offense_gate)
    tm_to_off_gate = np.linalg.norm(teammates_pos - offense_gate, axis=1)
    opp_to_off_gate = np.linalg.norm(opponents_pos - offense_gate, axis=1)
    
    rank_s_off_gate_tm = np.sum(tm_to_off_gate < s_to_off_gate) + 1 
    rank_s_off_gate_opp = np.sum(opp_to_off_gate < s_to_off_gate) + 1 
    
    top_sideline_y = PITCH_Y_MAX
    s_to_top = abs(sender_pos[1] - top_sideline_y)
    tm_to_top = np.abs(teammates_pos[:, 1] - top_sideline_y)
    opp_to_top = np.abs(opponents_pos[:, 1] - top_sideline_y)
    
    rank_s_top_tm = np.sum(tm_to_top < s_to_top) + 1 
    rank_s_top_opp = np.sum(opp_to_top < s_to_top) + 1 
    
    return rank_s_off_gate_tm, rank_s_off_gate_opp, rank_s_top_tm, rank_s_top_opp

def calc_team_geometry(teammates_pos, offense_gate, defense_gate):
    """
    Team Shape / Macro-Geometry Calculator
    """
    tm_to_off_line = np.abs(teammates_pos[:, 0] - offense_gate[0])
    tm_to_def_line = np.abs(teammates_pos[:, 0] - defense_gate[0])
    tm_to_top = np.abs(teammates_pos[:, 1] - PITCH_Y_MAX)
    
    st_closest_off_line = np.min(tm_to_off_line) 
    st_median_off_line = np.median(tm_to_off_line) 
    
    sorted_def = np.sort(tm_to_def_line)
    st_closest_def_no_gk = sorted_def[1] if len(sorted_def) > 1 else sorted_def[0] 
    
    st_closest_top = np.min(tm_to_top) 
    st_median_top = np.median(tm_to_top) 
    st_closest_bottom = np.min(np.abs(teammates_pos[:, 1] - (-PITCH_Y_MAX))) 
    
    return st_closest_off_line, st_closest_def_no_gk, st_closest_top, st_closest_bottom, st_median_off_line, st_median_top

def calc_s_is_gk(sender_pos, teammates_pos, defense_gate):
    """
    Sender Role Identifier (GK Check)
    """
    s_to_def_gate = np.linalg.norm(sender_pos - defense_gate)
    tm_to_def_gate = np.linalg.norm(teammates_pos - defense_gate, axis=1)
    is_s_gk = 1 if s_to_def_gate == np.min(tm_to_def_gate) else 0 
    return is_s_gk

# =============================================================================
# RECEIVER FEATURE HELPERS
# =============================================================================

def calc_r_zones(rec_pos, side):
    """
    Receiver Pitch Zone Classifier
    """
    rec_x_norm = rec_pos[0] * side
    r_back = 1 if rec_x_norm < -1750 else 0 
    r_mid = 1 if -1750 <= rec_x_norm <= 1750 else 0 
    r_front = 1 if rec_x_norm > 1750 else 0 
    return r_back, r_mid, r_front

def calc_r_landmarks(rec_pos, offense_gate, defense_gate):
    """
    Receiver Landmark Distance Calculator
    """
    r_to_off_gate = np.linalg.norm(rec_pos - offense_gate) 
    r_to_def_gate = np.linalg.norm(rec_pos - defense_gate) 
    r_to_center = np.linalg.norm(rec_pos) 
    is_r_center_circle = 1 if r_to_center < 915 else 0 
    return r_to_off_gate, r_to_def_gate, r_to_center, is_r_center_circle

def calc_r_geometry(rec_pos, sender_pos, side):
    """
    Pass Geometry Calculator
    """
    norm_r_s_x_diff = (rec_pos[0] - sender_pos[0]) * side 
    is_r_off_dir = 1 if norm_r_s_x_diff > 0 else 0 
    abs_y_diff = abs(rec_pos[1] - sender_pos[1]) 
    return norm_r_s_x_diff, is_r_off_dir, abs_y_diff

def calc_r_pressure(rec_pos, r_tms_pos, r_opps_pos):
    """
    Receiver Context/Pressure Calculator
    """
    r_tm_dists = np.linalg.norm(r_tms_pos - rec_pos, axis=1)
    r_tm_dists = r_tm_dists[r_tm_dists > 0.1] 
    
    r_opp_dists = np.linalg.norm(r_opps_pos - rec_pos, axis=1)
    
    r_closest_tm = np.min(r_tm_dists) if len(r_tm_dists) > 0 else 10000 
    r_closest_3_tm = np.mean(np.sort(r_tm_dists)[:3]) if len(r_tm_dists) >= 3 else r_closest_tm 
    
    r_closest_opp = np.min(r_opp_dists) if len(r_opp_dists) > 0 else 10000 
    r_closest_3_opp = np.mean(np.sort(r_opp_dists)[:3]) if len(r_opp_dists) >= 3 else r_closest_opp 
    
    return r_closest_tm, r_closest_3_tm, r_closest_opp, r_closest_3_opp

def calc_r_rel_sender(rec_pos, sender_pos, r_tms_pos, r_opps_pos):
    """
    Sender-Receiver Triangulation Calculator
    """
    # Neighbors relative to sender
    if len(r_tms_pos) > 0:
        full_tm_dists = np.linalg.norm(r_tms_pos - rec_pos, axis=1)
        full_tm_dists[full_tm_dists < 0.1] = 999999 
        closest_tm_idx = np.argmin(full_tm_dists)
        r_closest_tm_to_sender = np.linalg.norm(r_tms_pos[closest_tm_idx] - sender_pos) 
    else:
        r_closest_tm_to_sender = 10000

    if len(r_opps_pos) > 0:
        r_opp_dists = np.linalg.norm(r_opps_pos - rec_pos, axis=1)
        closest_opp_idx = np.argmin(r_opp_dists)
        r_closest_opp_to_sender = np.linalg.norm(r_opps_pos[closest_opp_idx] - sender_pos) 
    else:
        r_closest_opp_to_sender = 10000
        
    return r_closest_tm_to_sender, r_closest_opp_to_sender

def calc_r_is_gk(rec_pos, r_tms_pos, defense_gate, is_rec_same_team, offense_gate):
    """
    Receiver Role Identifier (GK Check)
    """
    own_goal = defense_gate if is_rec_same_team else offense_gate
    r_is_gk = 1 if np.linalg.norm(rec_pos - own_goal) == np.min(np.linalg.norm(r_tms_pos - own_goal, axis=1)) else 0
    return r_is_gk

def calc_r_ranks(rec_pos, sender_pos, r_tms_pos, r_opps_pos, offense_gate, defense_gate, is_rec_same_team):
    """
    Receiver Relative Ranking Calculator
    """
    # Determine the target goal (Attacking gate)
    r_target_gate = offense_gate if is_rec_same_team else defense_gate
    
    # --- 1. Rank by Distance to Goal (How advanced are they?) ---
    r_tm_to_gate = np.linalg.norm(r_tms_pos - r_target_gate, axis=1)
    r_opp_to_gate = np.linalg.norm(r_opps_pos - r_target_gate, axis=1)
    r_dist_gate = np.linalg.norm(rec_pos - r_target_gate)
    
    rank_r_gate_tm = np.sum(r_tm_to_gate < r_dist_gate) + 1 
    rank_r_gate_opp = np.sum(r_opp_to_gate < r_dist_gate) + 1 
    
    # --- 2. Rank by Width (Top Sideline) ---
    top_sideline_y = PITCH_Y_MAX
    r_dist_top = abs(rec_pos[1] - top_sideline_y)
    r_tm_to_top = np.abs(r_tms_pos[:, 1] - top_sideline_y)
    r_opp_to_top = np.abs(r_opps_pos[:, 1] - top_sideline_y)
    
    rank_r_top_tm = np.sum(r_tm_to_top < r_dist_top) + 1 
    rank_r_top_opp = np.sum(r_opp_to_top < r_dist_top) + 1 

    # --- 3. Rank by Width (Bottom Sideline) ---
    bottom_sideline_y = -PITCH_Y_MAX
    r_dist_bottom = abs(rec_pos[1] - bottom_sideline_y)
    r_tm_to_bottom = np.abs(r_tms_pos[:, 1] - bottom_sideline_y)
    r_opp_to_bottom = np.abs(r_opps_pos[:, 1] - bottom_sideline_y)
    
    rank_r_bottom_tm = np.sum(r_tm_to_bottom < r_dist_bottom) + 1
    rank_r_bottom_opp = np.sum(r_opp_to_bottom < r_dist_bottom) + 1
    
    # --- 4. Rank by Proximity to Sender (Support options) ---
    dist_s_r = np.linalg.norm(sender_pos - rec_pos)
    r_tm_to_s = np.linalg.norm(r_tms_pos - sender_pos, axis=1)
    r_opp_to_s = np.linalg.norm(r_opps_pos - sender_pos, axis=1)
    
    rank_r_s_tm = np.sum(r_tm_to_s < dist_s_r) + 1 
    rank_r_s_opp = np.sum(r_opp_to_s < dist_s_r) + 1 
    
    return rank_r_gate_tm, rank_r_gate_opp, rank_r_top_tm, rank_r_top_opp, rank_r_bottom_tm, rank_r_bottom_opp, rank_r_s_tm, rank_r_s_opp

# =============================================================================
# FEATURE EXTRACTION CONTEXT & MAIN
# =============================================================================

def parse_match_context(row, side):
    """
    Match Snapshot Parser
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
    is_rec_same_team = 1 if (is_team1 and pid <= 11) or (not is_team1 and pid > 11) else 0 
    dist_s_r = np.linalg.norm(sender_pos - rec_pos) 
    
    # 1. Zones
    r_back, r_mid, r_front = calc_r_zones(rec_pos, side)
    
    # Check if in same field zone
    s_r_same_field = 1 if (sender_meta["s_back"]==r_back and sender_meta["s_mid"]==r_mid and sender_meta["s_front"]==r_front) else 0 
    
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
    # 1. Calculate the direct distance of the pass
    dist_s_r = np.linalg.norm(sender_pos - rec_pos)
    
    dists_to_line = []      # Store perpendicular distances of all opponents to the pass line
    dangerous_count = 0     # Counter for opponents physically between sender and receiver
    pass_angles = []        # Store angles between the pass vector and vectors to opponents
    
    # Vector representing the pass 
    vec_s_r = rec_pos - sender_pos
    
    for opp in opponents_pos:
        # Calculate the shortest distance from the opponent to the infinite line or segment connecting sender and receiver
        # This helps identify if an opponent is "blocking" the lane, even if they aren't exactly on the segment
        d_line = dist_point_to_segment(opp, sender_pos, rec_pos)
        dists_to_line.append(d_line)
        
        # Check if the opponent is within the "bounding box" or danger zone of the pass.
        # We use a simple heuristic: if the opponent is closer to the sender AND closer to the receiver
        # than the total pass distance, they are likely roughly "between" the two players.
        d_o_s = np.linalg.norm(opp - sender_pos) # Distance Opponent <-> Sender
        d_o_r = np.linalg.norm(opp - rec_pos)    # Distance Opponent <-> Receiver
        
        # If opponent is closer to both endpoints than the length of the pass, 
        # they form a triangle with the pass that implies they are somewhat in the middle.
        if d_o_s < dist_s_r and d_o_r < dist_s_r:
            dangerous_count += 1
            
            # Calculate the angle between the pass trajectory and the opponent.
            # Small angle = Opponent is directly ahead. Large angle = Opponent is to the side.
            vec_s_o = opp - sender_pos
            angle = get_angle_between_vectors(vec_s_r, vec_s_o)
            pass_angles.append(angle)
    
    # Sort distances to find the primary blockers (closest to the passing lane)
    dists_to_line.sort()
    
    # Get the distance of the 1st, 2nd, and 3rd closest opponents to the passing lane.
    # High values mean the lane is wide open. Low values mean tight passing windows.
    # Default to 100 (large distance) if fewer than 3 opponents exist.
    min_opp_dist_line = dists_to_line[0] if len(dists_to_line) > 0 else 100
    sec_opp_dist_line = dists_to_line[1] if len(dists_to_line) > 1 else 100
    thd_opp_dist_line = dists_to_line[2] if len(dists_to_line) > 2 else 100
    
    # Logic Check: If the receiver is NOT a teammate (i.e., we are calculating features for an opponent),
    # concepts like "passing lane safety" don't apply in the same way. 
    # We set a flag value (-1) to indicate this is an invalid/adversarial target.
    if not is_rec_same_team:
        min_opp_dist_line = -1
        
    # Find the smallest angle among dangerous opponents. Indicates the opponent most directly in the path of the ball.
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
        full_rec_feats = rec_feats[:-8] + path_feats + rec_feats[-8:]
        
        results.append(common_feats + full_rec_feats)
        
    return results

def make_dataset(X, y=None, sides=None):
    """
    Full Dataset Constructor
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
    """
    df = pd.read_csv(path, index_col=0)
    return df

def augment_data(X, y=None):
    """
    Dataset Augmentor (Mirroring)
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
    
    def __init__(self, n_models=5, estimators_config=None):
        """
        Ensemble Initialization
        """
        self.n_models = n_models
        self.estimators_config = estimators_config if estimators_config is not None else ['lgbm', 'rf', 'et']
        self.models = []
        self.feature_importances_ = None
        self.feature_names_ = None
        self.val_scores_history_ = []

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Ensemble Training Loop
        """
        self.models = []
        self.val_scores_history_ = []
        print(f"Training Bagged Ensemble ({self.n_models} generations)...")
        print(f"Base Estimators: {self.estimators_config}")
        
        SAFE_N_JOBS = 4

        

        for i in range(self.n_models):
            seed = 42 + i
            
            # Define Base Learners Dictionary
            base_learners_dict = {
                'lgbm': LGBMClassifier(
                n_estimators=180,
                max_depth=9,
                num_leaves=50,
                learning_rate=0.04,
                objective='binary',
                scale_pos_weight=2.0,
                random_state=seed,
                n_jobs=SAFE_N_JOBS,
                verbose=-1
            ),

            'rf': RandomForestClassifier(
                n_estimators=300,
                max_depth=15,
                min_samples_leaf=4,
                random_state=seed,
                n_jobs=SAFE_N_JOBS 
            ),

            'et': ExtraTreesClassifier(
                n_estimators=250,
                max_depth=15,
                min_samples_leaf=4,
                random_state=seed,
                n_jobs=SAFE_N_JOBS 
            )
            }
            
            # Select only requested estimators
            estimators_list = [(name, base_learners_dict[name]) for name in self.estimators_config]
            
            # Stacking Classifier
            base_stacking = StackingClassifier(
                estimators=estimators_list, 
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
        """
        probs = sum(m.predict_proba(X) for m in self.models) / len(self.models)
        return probs

    def score_multiclass(self, X, y):
        """
        Multiclass Accuracy Scorer
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

def ensure_graph_dir(subfolder=None):
    """
    Directory Manager
    """
    path = GRAPH_OUTPUT_DIR
    if subfolder:
        path = os.path.join(GRAPH_OUTPUT_DIR, subfolder)
    if not os.path.exists(path): os.makedirs(path)
    return path

def plot_feature_importance(model, filename="feature_importance.png", folder=None):
    """
    Feature Importance Plotter
    """
    if model.feature_importances_ is None: return
    out_dir = ensure_graph_dir(folder)
    df_imp = pd.DataFrame({'feature': model.feature_names_, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=df_imp.head(30), hue='feature', palette='viridis', legend=False)
    plt.title('Top 30 Advanced Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

def plot_prediction_confidence(probas, filename="confidence_distribution.png", folder=None):
    """
    Confidence Distribution Plotter
    """
    out_dir = ensure_graph_dir(folder)
    max_probs = np.max(probas, axis=1)
    plt.figure(figsize=(10, 6))
    sns.histplot(max_probs, bins=50, kde=True, color='purple')
    plt.title('Prediction Confidence Distribution (After Softmax)')
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

def plot_fold_scores(scores, filename="learning_curve.png", folder=None):
    """
    Validation Scores per Fold Plotter (Renamed from plot_learning_curve)
    """
    if not scores: return
    out_dir = ensure_graph_dir(folder)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(scores) + 1), scores, marker='o', color='b')
    plt.title('Validation Accuracy per Fold (10-Fold CV)')
    plt.xlabel('Fold Number')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

def plot_per_player_precision(precisions_list, filename="player_precision_boxplot.png", folder=None):
    """
    Player Precision Boxplotter
    """
    if not precisions_list: return
    out_dir = ensure_graph_dir(folder)
    # Convert list of arrays to DataFrame
    df = pd.DataFrame(precisions_list, columns=[f'P{i}' for i in range(1, 23)])
    df_melted = df.melt(var_name='Player', value_name='Precision')
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Player', y='Precision', data=df_melted, palette='coolwarm', hue='Player', legend=False)
    plt.title('Precision Distribution per Player across Folds')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png", folder=None):
    """
    Confusion Matrix Heatmap Plotter
    """
    out_dir = ensure_graph_dir(folder)
    cm = confusion_matrix(y_true, y_pred, labels=range(22))
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'P{i}' for i in range(1, 23)],
                yticklabels=[f'P{i}' for i in range(1, 23)])
    plt.xlabel('Predicted Player')
    plt.ylabel('True Receiver')
    plt.title('Aggregated Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

def plot_comparison_chart(results_dict):
    """
    Comparison Chart Plotter
    """
    out_dir = ensure_graph_dir()
    cases = list(results_dict.keys())
    scores = list(results_dict.values())
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=cases, y=scores, palette='viridis', hue=cases, legend=False)
    plt.title('Model Configuration Comparison (Validation Accuracy)')
    plt.ylabel('Mean Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_comparison.png"))
    plt.close()

def write_submission(predictions=None, probas=None, estimated_score=0, file_name="submission", date=True, indexes=None):
    """
    Kaggle Submission Writer
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

def run_experiment_case(case_name, X, y, n_original_passes, config, X_test=None, test_index=None):
    """
    Runs a single experimental case based on config parameters.
    Also generates a submission file for the test set.
    """
    print(f"\n>>> Running Case: {case_name}")
    print(f"    Config: {config}")
    
    fold_scores = []
    final_models_list = []
    
    # Containers for report graphs
    fold_precisions = []
    all_val_trues = []
    all_val_preds = []
    
    # Accumulator for Test Set Predictions
    test_probs_sum = None
    
    use_kfold = config.get('use_kfold', True)
    n_bagging = config.get('n_bagging', 3)
    estimators = config.get('estimators', ['lgbm', 'rf', 'et'])
    
    if use_kfold:
        # 10-Fold CV Strategy
        pass_indices = np.arange(n_original_passes)
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        splits = list(kf.split(pass_indices))
    else:
        # Single Split Strategy (80/20)
        pass_indices = np.arange(n_original_passes)
        train_idx_passes, val_idx_passes = train_test_split(pass_indices, test_size=0.2, random_state=42)
        splits = [(train_idx_passes, val_idx_passes)]
        
    for fold_idx, (train_idx_passes, val_idx_passes) in enumerate(splits):
        if use_kfold:
            print(f"Fold {fold_idx+1}/10")
        else:
            print(f"Single Split (80/20)")
        
        # Expand indices to include mirrored versions (i + N)
        train_idx_extended = np.concatenate([train_idx_passes, train_idx_passes + n_original_passes])
        val_idx_extended = np.concatenate([val_idx_passes, val_idx_passes + n_original_passes])
        
        # Map Pass Indices -> Augmented Row Indices (x22)
        train_idx_final = (train_idx_extended[:, None] * 22 + np.arange(22)).flatten()
        val_idx_final = (val_idx_extended[:, None] * 22 + np.arange(22)).flatten()
        
        X_train_f = X.iloc[train_idx_final]
        y_train_f = y.iloc[train_idx_final]
        X_val_f = X.iloc[val_idx_final]
        y_val_f = y.iloc[val_idx_final]
        
        # Train Model
        model = BaggedEnsemble(n_models=n_bagging, estimators_config=estimators) 
        model.fit(X_train_f, y_train_f, X_val=X_val_f, y_val=y_val_f)
        final_models_list.append(model)
        
        # Score
        score = model.score_multiclass(X_val_f, y_val_f)
        fold_scores.append(score)
        print(f"  > Val Score: {score:.4f}")
        
        # Collect Validation Stats
        raw_probs = model.predict_proba(X_val_f)[:, 1]
        probs_2d = softmax(raw_probs.reshape(-1, 22))
        y_val_reshaped = y_val_f.values.ravel().reshape(-1, 22)
        
        val_preds_indices = np.argmax(probs_2d, axis=1)
        val_true_indices = np.argmax(y_val_reshaped, axis=1)
        
        per_class_prec = precision_score(val_true_indices, val_preds_indices, average=None, labels=range(22), zero_division=0)
        fold_precisions.append(per_class_prec)
        all_val_trues.extend(val_true_indices)
        all_val_preds.extend(val_preds_indices)
        
        # Accumulate TEST Set Predictions (if provided)
        if X_test is not None:
            # Predict raw probs for class 1
            fold_test_raw_probs = model.predict_proba(X_test)[:, 1]
            if test_probs_sum is None:
                test_probs_sum = fold_test_raw_probs
            else:
                test_probs_sum += fold_test_raw_probs
        
    avg_score = np.mean(fold_scores)
    print(f"  >>> Average Score for {case_name}: {avg_score:.4f}")
    
    # Graphs
    plot_fold_scores(fold_scores, "learning_curve.png", folder=case_name)
    plot_per_player_precision(fold_precisions, folder=case_name)
    plot_confusion_matrix(all_val_trues, all_val_preds, folder=case_name)
    plot_feature_importance(final_models_list[0], folder=case_name)
    
    # GENERATE SUBMISSION FILE FOR THIS CASE
    if X_test is not None and test_probs_sum is not None:
        print(f"  > Generating submission file for {case_name}...")
        avg_test_raw = test_probs_sum / len(splits)
        # Reshape and Softmax
        test_probs_2d = softmax(avg_test_raw.reshape(-1, 22))
        
        out_dir = ensure_graph_dir(case_name)
        # File name without extension (write_submission adds .csv)
        sub_path = os.path.join(out_dir, f"submission_{case_name}")
        write_submission(probas=test_probs_2d, estimated_score=avg_score, file_name=sub_path, date=False, indexes=test_index)
    
    return avg_score, final_models_list

def run_all_experiments(input_file, output_file):
    """
    Main Driver for Comparison Experiments
    """
    prefix = os.path.dirname(os.path.abspath(__file__)) + '/'
    X_original = load_data(prefix + input_file)
    y_original = load_data(prefix + output_file)
    
    # AUGMENTATION
    X_full, y_full = augment_data(X_original, y_original)
    s_full = detect_sides(X_full)

    # FEATURE GENERATION (Train)
    X_full_f, y_full_f = make_dataset(X_full, y_full, s_full)
    n_original_passes = len(X_original)
    
    # --- LOAD TEST SET FOR SUBMISSION GENERATION ---
    print("\nLoading Test Set for Submission Generation...")
    try:
        X_test_original = load_data(prefix + 'input_test_set.csv')
        s_test = detect_sides(X_test_original)
        # Generate features for test set (no targets)
        X_test_f, _ = make_dataset(X_test_original, None, s_test)
        test_index = X_test_original.index
    except FileNotFoundError:
        print("Warning: input_test_set.csv not found. Submissions will NOT be generated.")
        X_test_f = None
        test_index = None
    # -----------------------------------------------
    
    # TEST CASES DEFINITION
    test_cases = {
        "4_TunedParams": {
            'estimators': ['lgbm', 'rf', 'et'], 
            'use_kfold': True, 
            'n_bagging': 3
        }
    }
    
    results = {}
    
    for case_name, config in test_cases.items():
        score, models = run_experiment_case(case_name, X_full_f, y_full_f, n_original_passes, config, X_test=X_test_f, test_index=test_index)
        results[case_name] = score
        
    # Final Comparison Chart
    print("\n------------------------------------------------")
    print("FINAL EXPERIMENT RESULTS")
    print("------------------------------------------------")
    for k, v in results.items():
        print(f"{k:<20}: {v:.4f}")
        
    plot_comparison_chart(results)
    
    # Return best model list for potential submission
    best_case = max(results, key=results.get)
    print(f"\nBest Configuration: {best_case}")
    return results[best_case], [] # Returning empty model list as we are in experiment mode

if __name__ == '__main__':
    # Adjust filenames if they differ
    run_all_experiments('input_train_set.csv', 'output_train_set.csv')