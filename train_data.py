import chess
import game
import pandas as pd
import numpy as np
import time
import random

random.seed(42)  # For reproducibility


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

        game_state = create_game_state(moves)

        # Get the feature representation of the game state
        features = game_state.to_features() # features is a pd.DataFrame
        features['winner'] = 1 if row['winner'] == 'white' else 0

        # Append the features to the training data
        training_data = pd.concat([training_data, features], ignore_index=True)

        if (i + 1) % 100 == 0:
            print(f'Processed game {i + 1}/{len(game_data)}')

    training_data.reset_index(drop=True, inplace=True)
    return training_data



produce_training_data().to_csv('data/training_data.csv', index=False)



