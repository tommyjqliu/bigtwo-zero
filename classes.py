# It is recommended that you do not modify this file

from typing import List


class Trick:

    def __init__(self, playerNum: int, cards: List[str]):
        self.playerNum = (
            playerNum  # The player number of the player that played this trick.
        )
        self.cards = cards  # The cards that were played in this trick. E.g. ['3D', '5D', '9D', 'TD', 'KD']. Empty list if trick was passed.


class GameHistory:

    def __init__(
        self, finished: bool, winnerPlayerNum: int, gameHistory: List[List[Trick]]
    ):
        self.finished = finished  # Whether this game is finished. If false, then this is the game that is currently being played.
        self.winnerPlayerNum = winnerPlayerNum  # The player number of this game's winner, -1 if game has not yet finished.
        self.gameHistory = gameHistory  # A list of rounds, with each round consisting of a list of tricks. Each round will finish with 3 pass tricks unless a player won in that round.


class Player:

    def __init__(self, points: int, handSize: int):
        self.points = points  # The total number of points a player is currently on
        self.handSize = (
            handSize  # The number of cards in this player's hand in the current game
        )


class MatchState:

    def __init__(
        self,
        myPlayerNum: int,
        players: List[Player],
        myHand: List[str],
        toBeat: Trick | None,
        matchHistory: List[GameHistory],
        myData: str,
    ):
        self.myPlayerNum = myPlayerNum  # The player number of your algorithm
        self.players =  players  # Info about each player, indices will match player numbers
        self.myHand = myHand  # The cards currently in your hand. E.g. ['4D', '4C', '6D', '8H', 'TS', 'AD', '2S']
        self.toBeat = toBeat  # The trick that you must beat if you wish to play a trick
        self.matchHistory = matchHistory  # The game history of each game played so far, including the one currently being played, ordered from oldest to newest
        self.myData = myData  # A string that was generated by the previous iteration of getAction() to communicate with the curernt iteration
