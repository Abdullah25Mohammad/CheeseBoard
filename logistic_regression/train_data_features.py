import chess
import game
import pandas as pd
import numpy as np
import time
import random


game_data = pd.read_csv('data/games.csv')
game_data = game_data.dropna()
game_data = game_data[game_data["victory_status"] != "outoftime"] # remove games that ended by timeout
game_data = game_data[game_data["victory_status"] != "draw"] # remove games that ended in a draw
game_data = game_data[game_data["turns"] > 10] # remove games with less than 10 turns
game_data = game_data[["victory_status", "winner", "turns", "moves"]]

game_data = game_data.reset_index(drop=True)
# game_data.to_csv('game_data.csv', index=False)


def create_game_state(moves):
    """
    Create a game state from a list of moves.
    Move list are like "d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5 Bf4"
    """
    board = chess.Board()
    for move in moves.split():
        board.push_san(move)
        
    fen = board.fen()

    game_state = game.GameState.from_fen(fen)

    return game_state


def extract_features(state):
    """
    Extract features from the game state.
    """
    features = {
        'current_player': 0 if state.current_player == 1 else 1,  # 0 for white, 1 for black
        'turns_played': state.turns_played,
    }

    # count of each piece
    piece_counts = {}
    for piece in [1, 2, 3, 4, 5, 
                  -1, -2, -3, -4, -5]: # 0-5 for white pieces, -1 to -5 for black pieces, excluding 6 and -6 (kings)
        name = game.PIECE_NAMES[piece].replace(' ', '_').lower()  # convert to lowercase and replace spaces with underscores


        piece_counts[name] = np.count_nonzero(state.board == piece)
    
    features.update(piece_counts)


    # double pawns
    features['white_double_pawns'] = 0
    features['black_double_pawns'] = 0
    for file in range(8):
        white_pawn_count = np.count_nonzero(state.board[:, file] == 1)
        if white_pawn_count > 1:
            features['white_double_pawns'] += white_pawn_count - 1

        black_pawn_count = np.count_nonzero(state.board[:, file] == -1)
        if black_pawn_count > 1:
            features['black_double_pawns'] += black_pawn_count - 1

    # isolated pawns
    features['white_isolated_pawns'] = 0
    features['black_isolated_pawns'] = 0
    for file_i in range(8):
        file = state.board[:, file_i]
        if np.count_nonzero(file == 1) > 0:
            # check if there are pawns in adjacent files
            left_file = state.board[:, file_i - 1] if file_i > 0 else None
            right_file = state.board[:, file_i + 1] if file_i < 7 else None
            
            if (left_file is None or np.count_nonzero(left_file == 1) == 0) and \
               (right_file is None or np.count_nonzero(right_file == 1) == 0):
                features['white_isolated_pawns'] += np.count_nonzero(file == 1)


            if (left_file is None or np.count_nonzero(left_file == -1) == 0) and \
                (right_file is None or np.count_nonzero(right_file == -1) == 0):
                features['black_isolated_pawns'] += np.count_nonzero(file == -1)

    # passed pawns
    features['white_passed_pawns'] = 0
    features['black_passed_pawns'] = 0
    for file_i in range(8):
        file = state.board[:, file_i]
        if np.count_nonzero(file == 1) > 0:
            # check if there are pawns in adjacent files
            left_file = state.board[:, file_i - 1] if file_i > 0 else None
            right_file = state.board[:, file_i + 1] if file_i < 7 else None
            
            if (left_file is None or np.count_nonzero(left_file == -1) == 0) and \
               (right_file is None or np.count_nonzero(right_file == -1) == 0):
                features['white_passed_pawns'] += np.count_nonzero(file == 1)

            if (left_file is None or np.count_nonzero(left_file == 1) == 0) and \
                (right_file is None or np.count_nonzero(right_file == 1) == 0):
                features['black_passed_pawns'] += np.count_nonzero(file == -1)

    # mobility
    white_moves = game.get_all_valid_moves(state, 1)
    white_move_count = np.vectorize(len)(white_moves)
    features['white_mobility'] = np.sum(white_move_count)

    black_moves = game.get_all_valid_moves(state, -1)
    black_move_count = np.vectorize(len)(black_moves)
    features['black_mobility'] = np.sum(black_move_count)

    
    
    return features





# s = create_game_state("d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 Bb4+ Nc3 Ba5 Bf4")
# t = time.time()
# features = extract_features(s)
# print("Time taken to extract features:", time.time() - t)
# print("Extracted features:", features)




def produce_training_data():
    """
    Produce training data from the game data.
    """
    training_data = pd.DataFrame()

    for i, row in game_data.iterrows():
        moves = row['moves']

        # cut moves at a random point to simulate mid-game states
        move_list = moves.split()
        cut_index = random.randint(0, len(move_list) - 1)
        moves = ' '.join(move_list[:cut_index + 1])

        # create the game state from the moves
        state = create_game_state(moves)
        features = extract_features(state)
        
        # add the features to the training data
        features_df = pd.DataFrame([features])
        features_df['winner'] = {"white": 0, "black": 1, "draw": 2}[row['winner']]
        
        training_data = pd.concat([training_data, features_df], ignore_index=True)

        if i % 100 == 0:
            print(f"Processed {i}/{len(game_data)} games")

    return training_data


produce_training_data().to_csv('data/training_data.csv', index=False)



