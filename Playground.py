import ChessBot
model, move_vocab = ChessBot.train_chess_move_predictor()
move_generator = ChessBot.predict_move(model, "3Q4/5p2/1p3P2/PP1PrN2/5pq1/1k2p3/8/4KB1r w - - 0 1", move_vocab)
