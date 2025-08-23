import pandas as pd
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    """
    Dataset for chess position -> move prediction
    Input: FEN positions
    Output: Move strings (encoded as indices)
    """

    def __init__(self, df, move_vocab=None, is_training=True):
        super().__init__()

        self.is_training = is_training
        self.positions = df.iloc[:, 0].values  # First column: positions (FEN strings)

        if is_training and len(df.columns) > 1:
            self.moves = df.iloc[:, 1].values  # Second column: moves

            # Create move vocabulary if not provided
            if move_vocab is None:
                unique_moves = set(self.moves)
                # Reserve 0 for unknown moves
                self.move_vocab = {move: idx + 1 for idx, move in enumerate(sorted(unique_moves))}
                self.move_vocab['<UNK>'] = 0
            else:
                self.move_vocab = move_vocab

            # Encode moves
            self.move_indices = [self.move_vocab.get(move, 0) for move in self.moves]
        else:
            self.moves = None
            self.move_vocab = move_vocab if move_vocab else {}
            self.move_indices = None

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        # Convert FEN position to tensor
        position_tensor = self.fen_to_tensor(self.positions[idx])

        if self.is_training and self.move_indices is not None:
            move_idx = torch.tensor(self.move_indices[idx], dtype=torch.long)
            return position_tensor, move_idx
        else:
            return position_tensor

    def fen_to_tensor(self, fen_string):
        """Convert FEN string to 12-channel 8x8 tensor"""
        piece_map = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,  # White pieces
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11  # Black pieces
        }

        # Initialize 12-channel board
        board_tensor = torch.zeros(12, 8, 8, dtype=torch.float32)

        # Parse only the board part of FEN (before first space)
        board_part = fen_string.split()[0]
        rows = board_part.split('/')

        for row_idx, row in enumerate(rows):
            col_idx = 0
            for char in row:
                if char.isdigit():
                    col_idx += int(char)  # Skip empty squares
                else:
                    channel = piece_map.get(char, -1)
                    if channel >= 0 and col_idx < 8:
                        board_tensor[channel, row_idx, col_idx] = 1
                    col_idx += 1

        return board_tensor

    def save_vocab(self, filepath):
        """Save move vocabulary"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.move_vocab, f)

    @staticmethod
    def load_vocab(filepath):
        """Load move vocabulary"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def get_vocab_size(self):
        """Get vocabulary size for model initialization"""
        return len(self.move_vocab)