import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import NeuralNet
from ChessDataset import ChessDataset
from TrainTestSplit import TrainTestSplit as TTS
import torch.nn.functional as F


def train_chess_move_predictor(epochs=50, batch_size=32, lr=0.001):
    """
    Complete training function for position -> move prediction

    Args:
        train_df: DataFrame with columns ['CurrentPosition', 'Moves']
        test_df: Optional test DataFrame
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
    """

    print("Setting up datasets...")
    trainData, testData = TTS("./Documents/mygames.csv", "ngsterAHH")
    # Create training dataset
    train_dataset = ChessDataset(trainData)
    train_dataset.save_vocab('chess_vocab.pkl')

    # Create test dataset if provided

    vocab = ChessDataset.load_vocab('chess_vocab.pkl')
    test_dataset = ChessDataset(testData, move_vocab = vocab)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Vocabulary size: {train_dataset.get_vocab_size()}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create model
    model = NeuralNet.ChessMovePredictor(num_moves=train_dataset.get_vocab_size())

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_idx, (positions, moves) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            move_logits = model(positions)
            loss = criterion(move_logits, moves)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(move_logits.data, 1)
            total_predictions += moves.size(0)
            correct_predictions += (predicted == moves).sum().item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch + 1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        # Epoch statistics
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions

        print(f'Epoch {epoch + 1}/{epochs}:')
        print(f'  Train Loss: {avg_loss:.4f}')
        print(f'  Train Accuracy: {accuracy:.4f}')

        # Test evaluation
        if test_loader is not None:
            model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0

            with torch.no_grad():
                for positions, moves in test_loader:
                    move_logits = model(positions)
                    loss = criterion(move_logits, moves)

                    test_loss += loss.item()
                    _, predicted = torch.max(move_logits.data, 1)
                    test_total += moves.size(0)
                    test_correct += (predicted == moves).sum().item()

            test_accuracy = test_correct / test_total
            avg_test_loss = test_loss / len(test_loader)

            print(f'  Test Loss: {avg_test_loss:.4f}')
            print(f'  Test Accuracy: {test_accuracy:.4f}')

            scheduler.step(avg_test_loss)
            model.train()

        print('-' * 50)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': train_dataset.get_vocab_size(),
        'move_vocab': train_dataset.move_vocab
    }, 'chess_move_model.pth')

    print("Training completed! Model saved as 'chess_move_model.pth'")

    return model, train_dataset.move_vocab


def predict_move(model, position_fen, move_vocab, top_k=5):
    """
    Predict the best moves for a given position

    Args:
        model: Trained model
        position_fen: FEN string of the position
        move_vocab: Move vocabulary dictionary
        top_k: Number of top moves to return
    """
    model.eval()

    # Create reverse vocabulary for decoding
    reverse_vocab = {idx: move for move, idx in move_vocab.items()}

    # Create dummy dataset for encoding
    dummy_df = pd.DataFrame({'position': [position_fen]})
    dummy_dataset = ChessDataset(dummy_df, move_vocab=move_vocab, is_training=False)

    # Get position tensor
    position_tensor = dummy_dataset[0].unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        move_logits = model(position_tensor)
        move_probs = F.softmax(move_logits, dim=1)

        # Get top k moves
        top_probs, top_indices = torch.topk(move_probs, top_k)

        # Decode moves
        top_moves = []
        for i in range(top_k):
            move_idx = top_indices[0][i].item()
            move = reverse_vocab.get(move_idx, '<UNK>')
            prob = top_probs[0][i].item()
            top_moves.append((move, prob))

    return top_moves


