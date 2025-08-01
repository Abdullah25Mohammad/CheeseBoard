import chess
import numpy as np


def board_to_arr(board):
    """
    Convert a chess.Board object to a numpy 12 channel array representation.
    """
    channels = np.zeros((12, 8, 8), dtype=np.uint8)

    for file in range(8):
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)

            if piece is not None:
                if piece.color == chess.WHITE:
                    channels[piece.piece_type - 1, rank, file] = 1
                else:
                    channels[piece.piece_type + 5 - 1, rank, file] = 1

    return channels


# test
if __name__ == "__main__":
    board = chess.Board()
    print(board)
    arr = board_to_arr(board)
    print(arr)

    
    