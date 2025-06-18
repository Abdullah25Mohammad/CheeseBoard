import numpy as np
import pandas as pd

# Chess

PIECE_NAMES = {
    0: "Empty",
    1: "White Pawn",
    2: "White Knight",
    3: "White Bishop",
    4: "White Rook",
    5: "White Queen",
    6: "White King",
    -1: "Black Pawn",
    -2: "Black Knight",
    -3: "Black Bishop",
    -4: "Black Rook",
    -5: "Black Queen",
    -6: "Black King"
}

PIECE_TYPES = {
    0: "Empty",
    1: "Pawn",
    2: "Knight",
    3: "Bishop",
    4: "Rook",
    5: "Queen",
    6: "King"
}

PIECE_ICONS = {
    1: "♙",   # White Pawn
    2: "♘",   # White Knight
    3: "♗",   # White Bishop
    4: "♖",   # White Rook
    5: "♕",   # White Queen
    6: "♔",   # White King
   -1: "♟",   # Black Pawn
   -2: "♞",   # Black Knight
   -3: "♝",   # Black Bishop
   -4: "♜",   # Black Rook
   -5: "♛",   # Black Queen
   -6: "♚"    # Black King
}

    


class GameState:
    """
    Represents the state of a chess game. Shouldn't be modified directly.
    """


    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.current_player = 1 # 1 for White, -1 for Black
        self.turns_played = 0 # Number of turns played in the game

    def default():
        staring_board = np.array([
            [4, 2, 3, 5, 6, 3, 2, 4],
            [1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1,-1,-1,-1,-1,-1,-1,-1],
            [-4,-2,-3,-5,-6,-3,-2,-4]
        ], dtype=int)
        
        new_state = GameState()
        new_state.board = staring_board
        new_state.current_player = 1
        new_state.turns_played = 0


        return new_state
    

    def to_features(self):
        """
        Converts the current game state to a pd dataframe of features.
        Flattened board, current player, and turns played.
        """
        # 64-length array representing the board
        cells = self.board.flatten()

        # Construct the row as a dictionary
        data_row = {f'cell_{i}': val for i, val in enumerate(cells)}
        data_row['current_player'] = 1 if self.current_player == 1 else 0
        data_row['turns_played'] = self.turns_played

        return pd.DataFrame([data_row])

        



    def copy(self):
        """
        Returns a deep copy of the current game state.
        """
        new_state = GameState()
        new_state.board = np.copy(self.board)
        new_state.current_player = self.current_player
        return new_state
    

    def from_fen(fen):
        """
        Creates a GameState from a FEN string.
        """
        fen_maps = {
            "r": -4, "n": -2, "b": -3, "q": -5, "k": -6, "p": -1,
            "R": 4, "N": 2, "B": 3, "Q": 5, "K": 6, "P": 1
        }


        new_state = GameState()
        rows = fen.split(" ")[0].split("/")
        rows = rows[::-1]


        for i, row in enumerate(rows):
            j = 0
            for char in row:
                if char.isdigit():
                    j += int(char)

                else:
                    if char in fen_maps:
                        new_state.board[i, j] = fen_maps[char]
                        j += 1
                    else:
                        raise ValueError(f"Invalid character in FEN: {char}")

        new_state.current_player = 1 if fen.split(" ")[1] == "w" else -1
        new_state.turns_played = (int(fen.split(" ")[5])-1) * 2 + (1 if new_state.current_player == -1 else 0) # turns played is 2 * full turns + 1 if it's black's turn
        return new_state
    



    def __str__(self):
        board_str = "\n".join(" ".join(f"{piece:2}" for piece in row) for row in self.board)
        return f"Current Player: {'White' if self.current_player == 1 else 'Black'}\nBoard:\n{board_str}"
    
    def show(self):
        """
        Displays the current state of the game in a human-readable format.
        """
        mapped_board = np.vectorize(lambda x: PIECE_ICONS.get(x, " ."))(self.board)
        df = pd.DataFrame(mapped_board, columns=list("abcdefgh"), index=list(range(1, 9)))
        df.index.name = "Rank"
        df.columns.name = "File"

        print(df.sort_index(ascending=False))
        print(f"Current Player: {'White' if self.current_player == 1 else 'Black'}")
    

def move_piece(state, start_pos, end_pos):
    """
    Moves a piece from start_pos to end_pos on the board.
    """
    if state.board[start_pos] == 0:
        raise ValueError(f"No piece at the starting position: {start_pos}")
    
    new_state = state.copy()

    # check if the move is valid
    valid_moves = get_all_valid_moves_for_piece(new_state, start_pos)
    if end_pos not in valid_moves:
        raise ValueError(f"Invalid move from {start_pos} to {end_pos}. Valid moves are: {valid_moves}")
    
    
    # check if the piece belongs to the current player
    piece = new_state.board[start_pos]
    if piece * new_state.current_player <= 0:
        raise ValueError(f"Cannot move opponent's piece at {start_pos}. Current player: {'White' if new_state.current_player == 1 else 'Black'}")

    # Move the piece
    new_state.board[end_pos] = new_state.board[start_pos]
    new_state.board[start_pos] = 0

    # check if the move puts the current player in check
    # if is_in_check(new_state, new_state.current_player):
    #     raise ValueError(f"Move from {start_pos} to {end_pos} puts the current player in check.")

    # Switch the current player
    new_state.current_player *= -1

    # Increment the turn count
    new_state.turns_played += 1

    

    return new_state

def get_all_valid_moves_for_piece(state, piece_pos, care_for_check=True):
    """
    Returns a list of all valid moves for the piece at piece_pos.
    """
    piece = state.board[piece_pos]
    if piece == 0:
        return []  # No piece at this position
    
    valid_moves = []

    if piece * state.current_player <= 0:
        raise ValueError(f"Cannot get valid moves for opponent's piece at {piece_pos}. Current player: {'White' if state.current_player == 1 else 'Black'}")
        

    # PAWN MOVES
    if abs(piece) == 1:
        dir = 1 if piece > 0 else -1
        start_row = 1 if piece > 0 else 6

        # Move forward once
        forward_pos = (piece_pos[0] + dir, piece_pos[1])
        if 0 <= forward_pos[0] < 8 and state.board[forward_pos] == 0:

            new_state = simulate_move(state, piece_pos, forward_pos)
            if not care_for_check or not is_in_check(new_state, state.current_player):
                valid_moves.append(forward_pos)

        # Move forward twice from starting position
        if piece_pos[0] == start_row:
            forward_pos = (piece_pos[0] + 2 * dir, piece_pos[1])
            if 0 <= forward_pos[0] < 8 and state.board[forward_pos] == 0:

                new_state = simulate_move(state, piece_pos, forward_pos)
                if not care_for_check or not is_in_check(new_state, state.current_player):
                    valid_moves.append(forward_pos)

        # Capture diagonally
        for dx in [-1, 1]:
            capture_pos = (piece_pos[0] + dir, piece_pos[1] + dx)
            if 0 <= capture_pos[0] < 8 and 0 <= capture_pos[1] < 8:
                if state.board[capture_pos] * piece < 0:
                    new_state = simulate_move(state, piece_pos, capture_pos)
                    if not care_for_check or not is_in_check(new_state, state.current_player):
                        valid_moves.append(capture_pos)
    
    
    # KNIGHT MOVES
    if abs(piece) == 2:
        knight_moves = [
            (2, 1), (2, -1), (-2, 1), (-2, -1),
            (1, 2), (1, -2), (-1, 2), (-1, -2)
        ]
        for dx, dy in knight_moves:
            new_pos = (piece_pos[0] + dx, piece_pos[1] + dy)
            if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
                if state.board[new_pos] * piece == 0:  # Empty space
                    new_state = simulate_move(state, piece_pos, new_pos)
                    if not care_for_check or not is_in_check(new_state, state.current_player):
                        valid_moves.append(new_pos)
                elif state.board[new_pos] * piece < 0:  # Capture opponent's piece
                    new_state = simulate_move(state, piece_pos, new_pos)
                    if not care_for_check or not is_in_check(new_state, state.current_player):
                        valid_moves.append(new_pos)


    # BISHOP MOVES
    if abs(piece) == 3:
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in directions:
            for step in range(1, 8):
                new_pos = (piece_pos[0] + step * dx, piece_pos[1] + step * dy)
                if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
                    if state.board[new_pos] == 0:
                        new_state = simulate_move(state, piece_pos, new_pos)
                        if not care_for_check or not is_in_check(new_state, state.current_player):
                            valid_moves.append(new_pos)
                    elif state.board[new_pos] * piece < 0: # Capture opponent's piece
                        new_state = simulate_move(state, piece_pos, new_pos)
                        if not care_for_check or not is_in_check(new_state, state.current_player):
                            valid_moves.append(new_pos)
                        break
                    else: # Blocked by own piece
                        break


    # ROOK MOVES
    if abs(piece) == 4:
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        for dx, dy in directions:
            for step in range(1, 8):
                new_pos = (piece_pos[0] + step * dx, piece_pos[1] + step * dy)
                if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
                    if state.board[new_pos] == 0:
                        new_state = simulate_move(state, piece_pos, new_pos)
                        if not care_for_check or not is_in_check(new_state, state.current_player):
                            valid_moves.append(new_pos)
                    elif state.board[new_pos] * piece < 0: # Capture opponent's piece
                        new_state = simulate_move(state, piece_pos, new_pos)
                        if not care_for_check or not is_in_check(new_state, state.current_player):
                            valid_moves.append(new_pos)
                        break
                    else: # Blocked by own piece
                        break

    # QUEEN MOVES
    if abs(piece) == 5:
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1),
                      (1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dx, dy in directions:
            for step in range(1, 8):
                new_pos = (piece_pos[0] + step * dx, piece_pos[1] + step * dy)
                if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
                    if state.board[new_pos] == 0:
                        new_state = simulate_move(state, piece_pos, new_pos)
                        if not care_for_check or not is_in_check(new_state, state.current_player):
                            valid_moves.append(new_pos)
                    elif state.board[new_pos] * piece < 0: # Capture opponent's piece
                        new_state = simulate_move(state, piece_pos, new_pos)
                        if not care_for_check or not is_in_check(new_state, state.current_player):
                            valid_moves.append(new_pos)
                        break
                    else: # Blocked by own piece
                        break

    # KING MOVES
    if abs(piece) == 6:
        king_moves = [
            (1, 0), (0, 1), (-1, 0), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        for dx, dy in king_moves:
            new_pos = (piece_pos[0] + dx, piece_pos[1] + dy)
            if 0 <= new_pos[0] < 8 and 0 <= new_pos[1] < 8:
                if state.board[new_pos] * piece == 0:  # Empty space
                    new_state = simulate_move(state, piece_pos, new_pos)
                    if not care_for_check or not is_in_check(new_state, state.current_player):
                        valid_moves.append(new_pos)
                elif state.board[new_pos] * piece < 0:  # Capture opponent's piece
                    new_state = simulate_move(state, piece_pos, new_pos)
                    if not care_for_check or not is_in_check(new_state, state.current_player):
                        valid_moves.append(new_pos)

    return valid_moves

def get_all_valid_moves(state, current_player):
    """
    Returns a 2d np array of lists of valid moves for each piece on the board.
    """
    valid_moves = np.empty((8, 8), dtype=object)

    for i in range(8):
        for j in range(8):
            piece_pos = (i, j)
            piece = state.board[piece_pos]
            if piece * current_player > 0:  # Only consider pieces of the current player
                valid_moves[piece_pos] = get_all_valid_moves_for_piece(state, piece_pos)
            else:
                valid_moves[piece_pos] = []


    return valid_moves


def simulate_move(state, start_pos, end_pos):
    """
    Simulates a move without modifying the original state.
    Returns a new GameState with the move applied.
    """
    if state.board[start_pos] == 0:
        raise ValueError(f"No piece at the starting position: {start_pos}")
    
    new_state = state.copy()

    # Move the piece
    new_state.board[end_pos] = new_state.board[start_pos]
    new_state.board[start_pos] = 0

    # Switch the current player
    new_state.current_player *= -1

    # Increment the turn count
    new_state.turns_played += 1

    return new_state


def is_in_check(state, player):
    """
    Checks if the current player is in check.
    """
    if player not in [1, -1]:
        raise ValueError("Invalid player. Player must be 1 (White) or -1 (Black).")
    king_pos = None
    for i in range(8):
        for j in range(8):
            if state.board[i, j] == 6 * player:  # King of the current player
                king_pos = (i, j)
                break
        if king_pos:
            break

    if not king_pos:
        return False  # No king found, should not happen in a valid game

    # Check if any opponent's piece can attack the king
    opponent = -player
    for i in range(8):
        for j in range(8):
            if state.board[i, j] * opponent > 0:  # Opponent's piece
                valid_moves = get_all_valid_moves_for_piece(state, (i, j), care_for_check=False)
                if king_pos in valid_moves:
                    return True  # King is in check

    return False  # King is not in check


class Game:
    """
    Represents a chess game. Shouldn't be modified directly.
    """

    def __init__(self):
        self.state = GameState.default()

    def move(self, start_pos, end_pos):
        """
        Moves a piece from start_pos to end_pos on the board.
        """
        self.state = move_piece(self.state, start_pos, end_pos)


    def show(self):
        """
        Displays the current state of the game.
        """
        mapped_state = np.vectorize(lambda x: PIECE_ICONS.get(x, " ."))(self.state.board)
        df = pd.DataFrame(mapped_state, columns=list("abcdefgh"), index=list(range(1, 9)))
        df.index.name = "Rank"
        df.columns.name = "File"

        print(df.sort_index(ascending=False))
        print(f"Current Player: {'White' if self.state.current_player == 1 else 'Black'}")





def gameloop():
    g = Game()

    while True:
        g.show()
        move = input("Enter your move: ").strip()
        if move.lower() == "exit":
            break

        try:
            start, end = move.split("-")
            start_pos = (int(start[1])-1, ord(start[0]) - ord('a'))  # Convert to (row, col)
            end_pos = (int(end[1])-1, ord(end[0]) - ord('a'))  # Convert to (row, col)

            g.move(start_pos, end_pos)
        except ValueError as e:
            print(f"Error: {e}")
        # except Exception as e:
        #     print(f"An unexpected error occurred: {e}")

        print()

if __name__ == "__main__":
    gameloop()
    pass

    # g = Game()
    # g.show()

    # next_states = get_all_valid_next_states_for_piece(g.state, (1, 0))  # Example: Get valid next states for the white pawn at (1, 0)
    # print(f"{len(next_states)} valid next states")

    # print()
    # print()

    # for i, state in enumerate(next_states):
    #     state.show()
    #     print()
    #     print("-" * 40)
    #     print()
    #     print()


        