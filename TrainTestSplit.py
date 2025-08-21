import pandas as pd


def TrainTestSplit(csv_path, username):
    data = pd.read_csv(csv_path)
    white_data = data[data["White"] == username]
    black_data = data[data["Black"] == username]
    white_data = white_data[["CurrentPosition", "whitemoves"]]
    black_data = black_data[["CurrentPosition", "blackmoves"]]
    white_data = white_data.rename(columns={"CurrentPosition": "CurrentPosition", "whitemoves": "Moves"})
    black_data = black_data.rename(columns={"CurrentPosition": "CurrentPosition", "blackmoves": "Moves"})
    cleanData = pd.concat([white_data, black_data])
    print(cleanData.head())
    train_data = cleanData.sample(frac = 0.9)
    test_data = cleanData.drop(train_data.index)
    return train_data, test_data