import game
import numpy as np

POS_MATRIX = np.array([[(x, y) for y in range(8)] for x in range(8)])

def get_all_valid_next_states_for_piece(state, piece_pos):
    """
    Returns a list of all valid next game states for the piece at piece_pos.
    """
    valid_moves = game.get_all_valid_moves_for_piece(state, piece_pos)
    next_states = []

    for move in valid_moves:
        new_state = game.move_piece(state, piece_pos, move)
        next_states.append(new_state)

    return next_states, valid_moves


def get_all_valid_next_states(state):
    """
    Returns a list of all valid next game states.
    """
    next_states = []
    
    # for piece_pos in state.get_all_piece_positions():
    #     next_states.extend(get_all_valid_next_states_for_piece(state, piece_pos))

    # vectorize operation and apply to all squares on the board

    sparse_pos = np.where(state.board != 0, POS_MATRIX, None)

    def get_next_states_for_square(pos):
        if pos is None:
            return []
        
        piece = state.board[pos]
        if piece == 0:
            return []
        
        piece_pos = (pos[0], pos[1])

        next_states, valid_moves = get_all_valid_next_states_for_piece(state, piece_pos)
        
        if not next_states:
            return []

        

    
    return next_states


state = game.GameState()
print(get_all_valid_next_states(state))