import re
import pandas as pd
import os
import requests
import json
import urllib.request
from pathlib import Path
import certifi
import builtins

class chessGameCSVExtractor:
    def __init__(self, tgtFilePath, PGNDirectory, user):
        self.pgnMeta = ["Event", "Site", "Date", "Round", "White", "Black", "Result",
                        "CurrentPosition", "Timezone", "ECO", "ECOURL", "UTDate", "UTCTime", "WhiteELO",
                        "BlackELO", "Timecontrol", "Termination", "StartTime", "EndDate", "EndTime", "Link",
                        "Moves"]  #metadata
        self.tgtFilePath = tgtFilePath  #This is the path where the final CSV gets created ("./Documents/mygames.csv")
        self.moveStartLine = 22  #moves start from line 22 in chess.com PGNs
        self.PGNDirectory = PGNDirectory  #This is the location where the API downloads the PGNs from the archives ("./Documents/PGN")
        self.user = user

    def getPGN(self):
        """This function accesses the chess.com public API and downloads all the PGNs to a folder"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python script for educational purposes'
        } #cheeky bit of fake headering to bypass bot restrictions
        certificate = certifi.where()
        pgn_archive_links = requests.get("https://api.chess.com/pub/player/" + self.user + "/games/archives", headers = headers, verify = certificate)

        for url in json.loads(pgn_archive_links.content)["archives"]:
            filepath = self.PGNDirectory + "/" + url.split("/")[7] + url.split("/")[8] + '.pgn'
            my_file = Path(filepath)
            if not my_file.is_file():
                urllib.request.urlretrieve(url + '/pgn', filepath)
            print("PGN SAVING!")

    def importPGNData(self, filepath):
        """This function returns the data read as a string"""
        with open(filepath) as f:
            return f.readlines()

    def getEdgePoints(self, data):
        """This function returns the start and end indices for each game in the PGN"""
        ends = []
        starts = []
        for n, l in enumerate(data):
            if l.startswith("[Event"):
                if n != 0:
                    ends.append(n - 1)
                starts.append(n)
            elif (n == len(data) - 1):
                ends.append(n)

        return (starts, ends)

    def grpGames(self, data, starts, ends):
        """This function groups games into individual lists based on the start and end index"""
        blocks = []
        for i in range(len(ends)):
            try:
                element = data[starts[i]: ends[i] + 1]
            except:
                print(i)
            if element not in blocks: blocks.append(element)
        return blocks

    def mergeMoves(self, game):
        """This function cleans out the moves and other attributes, removes newlines and formats the list to be converted into a dictionary"""
        firstmove = lastmove = -1
        for n, eachrow in enumerate(game):
            game[n] = str(game[n]).replace('\n', '')
            try:
                if n <= self.moveStartLine - 2: game[n] = self.stripwhitespace(game[n]).split('~')[1].strip(']["')
            except:
                if n <= self.moveStartLine - 4: game[n] = self.stripwhitespace(game[n]).split('~')[1].strip(']["')
                pass
        return list(filter(None, game))

    def stripwhitespace(self, text):
        lst = text.split('"')
        for i, item in enumerate(lst):
            if not i % 2:
                lst[i] = re.sub("\s+", "~", item)
        return '"'.join(lst)

    def createGameDictLetsPlay(self, game_dict):
        """This is a helper function to address games under Lets Play events on chess.com. These events have a slightly different way of representation than the Live Chess events"""
        for n, move in enumerate(game_dict["Moves"].split(" ")):

            if n % 3 == 0:  # every 3rd element is the move number
                if move == '1-0' or move == '0-1' or move == '1/2-1/2':
                    None
                else:
                    movenum = n
            elif n == movenum + 2:
                if move == '1-0' or move == '0-1' or move == '1/2-1/2':
                    None
                else:
                    game_dict["whitemoves"].append(move)
            else:
                if move == '1-0' or move == '0-1' or move == '1/2-1/2':
                    None
                else:
                    game_dict["blackmoves"].append(move)

        if len(game_dict["blackmoves"]) > len(game_dict["whitemoves"]): game_dict["whitemoves"].append("over")
        if len(game_dict["blackmoves"]) < len(game_dict["whitemoves"]): game_dict["blackmoves"].append("over")
        del game_dict["Moves"]
        return game_dict

    def createGameDictLiveChess(self, game_dict):
        """This is a helper function to address games under Live Chess events on chess.com."""
        try:
            for n, move in enumerate(game_dict["Moves"].split(" ")):

                if '{' in move or '}' in move:
                    None
                elif '.' in move:
                    movenum = int(move.split(".")[0])
                    print(movenum)
                    if "..." in move:
                        color = 'black'
                    else:
                        color = "white"
                else:
                    if color == "white":
                        if move == '1-0' or move == '0-1' or move == '1/2-1/2':
                            None
                        else:
                            game_dict["whitemoves"].append(move)
                    else:
                        if move == '1-0' or move == '0-1' or move == '1/2-1/2':
                            None
                        else:
                            game_dict["blackmoves"].append(move)

            if len(game_dict["blackmoves"]) > len(game_dict["whitemoves"]): game_dict["whitemoves"].append("over")
            if len(game_dict["blackmoves"]) < len(game_dict["whitemoves"]): game_dict["blackmoves"].append("over")
            del game_dict["Moves"]
        except:
            pass

        return game_dict

    def createGameDict(self, games):
        allgames = []
        for gamenum, eachgame in enumerate(games):
            game_dict = dict(zip(self.pgnMeta, eachgame))
            movenum = 0
            game_dict["whitemoves"] = []
            game_dict["blackmoves"] = []
            color = "white"
            if game_dict["Event"] == "Let's Play!":
                allgames.append(self.createGameDictLetsPlay(game_dict))
            else:
                allgames.append(self.createGameDictLiveChess(game_dict))

        return allgames

    def main(self):
        self.getPGN()
        tgtFilePathObj = Path(self.tgtFilePath)
        tgtFilePathObj.unlink(missing_ok=True)

        with os.scandir(self.PGNDirectory) as pgndir:
            for file in pgndir:
                print('*', end=" ")
                data = self.importPGNData(file)

                starts, ends = self.getEdgePoints(data)
                games = self.grpGames(data, starts, ends)
                print(f"list is: {list}")
                print(f"list type: {type(list)}")
                print(f"list id: {id(list)}")
                print(f"Is list callable? {callable(list)}")

                # Also check if it's in globals/locals
                print(f"Built-in list: {builtins.list}")
                print(f"Are they the same? {list is builtins.list}")
                games = list(map(self.mergeMoves, games))
                allgames = self.createGameDict(games)

                for gamenum, game in enumerate(allgames):
                    df = pd.DataFrame(allgames[gamenum])

                    with open(self.tgtFilePath, 'a') as f:
                        df.to_csv(f, mode='a', header=f.tell() == 0)
        print("Export Complete!")
