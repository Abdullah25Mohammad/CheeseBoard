import game
import numpy as np
from tf_keras.models import load_model
import time

MODEL = load_model("chess_cnn_model.h5")

POS_MATRIX = np.empty((8, 8), dtype=object)
for x in range(8):
    for y in range(8):
        POS_MATRIX[x, y] = (x, y)

class GameNode:
    """
    Represents a node in the game tree.
    """
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.moves = [] # moves leading to this state

    def get_value(self, model=MODEL):
        """
        Returns the value of the node based on the model's prediction.
        """
        # Assuming model.predict returns a value for the state
        return model(self.state.channelize()).numpy()[0][0]
        
    def extend(self):
        """
        Generates all valid next states from the current state and creates child nodes.
        Extends the tree by one level.
        """
        
        next_states, next_moves = flatten_next_states(get_all_valid_next_states(self.state))

        for next_state, move in zip(next_states, next_moves):
            child_node = GameNode(next_state, parent=self)
            child_node.moves = self.moves + [move]
            self.children.append(child_node)

            # print(f"Extended node with move {move} to state {next_state}")

        

    def recursive_extend(self, depth):
        """
        Recursively extends the tree to a given depth.
        """
        if depth == 0:
            return
        
        self.extend()
        
        for child in self.children:
            child.recursive_extend(depth - 1)


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

    # sparse_pos = np.where(state.board != 0, POS_MATRIX, None)
    # remove enemy pieces and empty squares
    sparse_pos = np.where(state.board * state.current_player > 0, POS_MATRIX, None)

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
        
        return list(zip(valid_moves, next_states))
        
    
    vectorized_get_next_states = np.vectorize(get_next_states_for_square, otypes=[object])
    move_states = vectorized_get_next_states(sparse_pos)

    
    return move_states

def flatten_next_states(move_states):
    """
    Flattens the list of move states into a single np array of next states.
    """
    next_states = []
    next_moves = []
    for x in range(8):
        for y in range(8):
            states_list = move_states[x, y]
            if len(states_list) == 0:
                continue

            # the states_list is a list of tuples (valid_move, next_state)

            # split this into two lists
            valid_moves, valid_states = zip(*states_list)
            next_states.extend(valid_states)

            def full_move(move):
                return ((x, y), move)
            
            next_moves.extend(map(full_move, valid_moves))
            

    return np.array(next_states), np.array(next_moves)


class Engine:
    """
    Represents the chess engine.
    """
    def __init__(self, game_state=None):
        # if game_state is None:
        #     self.game_state = game.GameState.default()
        # else:
        #     self.game_state = game_state

        # self.root_node = GameNode(self.game_state)
        # self.current_node = self.root_node
        pass

    def get_best_move(self, depth=3):
        """
        Returns the best move for the current game_node
        """
        node = GameNode(self.game_state)




# print(MODEL(state.channelize()).numpy())

# state = np.random.choice(flatten_next_states(get_all_valid_next_states(state))[0])
# print(MODEL(state.channelize()).numpy())

# state = np.random.choice(flatten_next_states(get_all_valid_next_states(state))[0])
# print(MODEL(state.channelize()).numpy())






# state = game.GameState.default()
# root_node = GameNode(state)
# start = time.time()
# root_node.recursive_extend(3)
# end = time.time()
# print(f"Time taken to extend tree: {end - start} seconds")

