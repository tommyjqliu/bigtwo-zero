from classes import *
import json
import numpy as np
from itertools import combinations, product

initial_cache = {}

class Algorithm:
    def getAction(self, state: MatchState):
        cache = initial_cache
        try:
            cache = load_dict(state.myData)
        except:
            pass
        gameHistory = state.matchHistory[-1].gameHistory
        all_played = [t for r in gameHistory for t in r]
        code_point = cards_to_code_point(state.myHand)
        actions = get_possible_actions(code_point)
        if not state.toBeat is None:
            action_in_charge = trick2action(state.toBeat)
            actions = get_legal_actions(actions, action_in_charge)
        elif len(all_played) == 0:  # first hand
            actions = [a for a in actions if a.code[0]]

        return code_point_to_cards(actions[-1].code), store_dict(cache)


ranks = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2"]
suits = ["D", "C", "H", "S"]


def code_point_to_cards(code_point):
    cards = []
    for i in range(52):
        if code_point[i]:
            rank_index = i // 4
            suit_index = i % 4
            cards.append(ranks[rank_index] + suits[suit_index])
    return cards


def cards_to_code_point(cards):
    code_point = np.zeros(52, dtype=bool)
    for card in cards:
        rank = card[0]
        suit = card[1]
        rank_index = ranks.index(rank)
        suit_index = suits.index(suit)
        index = rank_index * 4 + suit_index
        code_point[index] = True
    return code_point


def trick2action(trick: Trick):
    code_point = cards_to_code_point(trick.cards)
    if np.sum(code_point) == 5:
        s_sum = code_point.reshape(13, 4).sum(axis=1)
        r_sum = code_point.reshape(13, 4).T.sum(axis=1)
        trick_type = None
        critical_card = None
        is_fullhouse = np.any(s_sum == 3)
        is_fourkind = np.any(s_sum == 4)
        if is_fullhouse:
            rank = np.where(s_sum == 3)[0][0]
            critical_card = get_max_index(code_point[rank * 4 : (rank + 1) * 4])
            critical_card = rank * 4 + critical_card
            trick_type = 2
            return Action(code_point, critical_card, trick_type)
        if is_fourkind:
            rank = np.where(s_sum == 4)[0][0]
            critical_card = get_max_index(code_point[rank * 4 : (rank + 1) * 4])
            critical_card = rank * 4 + critical_card
            trick_type = 3
            return Action(code_point, critical_card, trick_type)

        critical_card = get_max_index(code_point)
        is_straight = stright_matrix[
            np.sum(np.bitwise_and(stright_matrix, s_sum != 0), axis=1) == 5
        ].any()
        is_flush = np.any(r_sum == 5)
        if is_straight and is_flush:
            return Action(code_point, critical_card, 4)
        elif is_flush:
            return Action(code_point, critical_card, 1)
        else:
            return Action(code_point, critical_card, 0)
    else:
        critical_card = get_max_index(code_point)
        return Action(code_point, critical_card, 0)


def store_dict(data):
    serializable_dict = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in data.items()
    }
    json_str = json.dumps(serializable_dict)
    return json_str


def load_dict(json_str):
    data = json.loads(json_str)

    loaded_dict = {
        key: np.array(value) if isinstance(value, list) else value
        for key, value in data.items()
    }
    return loaded_dict


class Action:
    def __init__(self, code=np.zeros(52, dtype=bool), critical_card=0, trick_type=0):
        self.count = np.sum(code)
        self.is_pass = self.count == 0
        self.code = code
        self.type = trick_type
        self.critical_card = critical_card


def get_possible_actions(code_point):
    s_code_point = code_point.reshape(13, 4)
    s_sum = s_code_point.sum(axis=1)
    r_code_point = code_point.reshape(13, 4).T
    r_sum = r_code_point.sum(axis=1)
    actions = []
    possible_2 = {}
    possible_3 = {}
    strightflush = np.empty((0, 52), dtype=bool)

    for i in range(52):
        # single
        if code_point[i]:
            actions.append(Action(np.eye(52, dtype=bool)[i], i, 0))

    for i in range(13):
        # pair
        if s_sum[i] >= 2:
            possibles = pair_matrix[
                np.sum(np.bitwise_and(pair_matrix, np.array(s_code_point[i])), axis=1)
                == 2
            ]
            possible_2[i] = possibles
            for p in possibles:
                code = np.zeros(52, dtype=bool)
                code[i * 4 : (i + 1) * 4] = p
                max_index = i * 4 + get_max_index(p)
                actions.append(Action(code, max_index, 0))

        # triplet
        if s_sum[i] >= 3:
            possibles = triplet_matrix[
                np.sum(
                    np.bitwise_and(triplet_matrix, np.array(s_code_point[i])), axis=1
                )
                == 3
            ]
            possible_3[i] = possibles
            for p in possibles:
                code = np.zeros(52, dtype=bool)
                code[i * 4 : (i + 1) * 4] = p
                max_index = i * 4 + get_max_index(p)
                actions.append(Action(code, max_index, 0))

    # straightflush
    for i in range(4):
        if r_sum[i] >= 5:
            possible_s = stright_matrix[
                np.sum(np.bitwise_and(stright_matrix, r_code_point[i]), axis=1) == 5
            ]
            for p in possible_s:
                code = np.zeros(52, dtype=bool)
                positions = np.where(p)[0] * 4 + i
                code[positions] = True
                max_index = positions[-1]
                actions.append(Action(code, max_index, 4))
                strightflush = np.vstack((strightflush, code))

    # straight
    possible_s = stright_matrix[
        np.sum(np.bitwise_and(stright_matrix, s_sum != 0), axis=1) == 5
    ]
    for p in possible_s:
        cs = []
        for i in range(13):
            if p[i]:
                crow = []
                for j, v in enumerate(s_code_point[i]):
                    if v:
                        c = np.zeros(52, dtype=bool)
                        c[i * 4 + j] = True
                        crow.append(c)
                cs.append(crow)

        for c in product(*cs):
            code = np.sum(c, axis=0, dtype=bool)
            if not np.any(np.all(strightflush == code, axis=1)):
                actions.append(Action(code, get_max_index(code), 0))

    # Flush
    for i in range(4):
        if r_sum[i] >= 5:
            t = np.where(r_code_point[i] > 0)[0]
            for indices in list(combinations(t, 5)):
                code = np.zeros(52, dtype=bool)
                positions = i + np.array(indices) * 4
                code[positions] = True
                if not np.any(np.all(strightflush == code, axis=1)):
                    max_index = positions[-1]
                    actions.append(Action(code, max_index, 1))

    # fullhouse
    for i in possible_3:
        for j in possible_2:
            if i != j:
                for p3 in possible_3[i]:
                    for p2 in possible_2[j]:
                        code = np.zeros(52, dtype=bool)
                        positions = i * 4 + np.where(p3)[0]
                        code[positions] = True
                        max_index = positions[-1]
                        positions = j * 4 + np.where(p2)[0]
                        code[positions] = True
                        actions.append(Action(code, max_index, 2))

    # fourkind
    for i in range(13):
        if s_sum[i] >= 4:
            for j in range(52):
                if code_point[j] and j // 4 != i:
                    code = np.zeros(52, dtype=bool)
                    positions = i * 4 + np.where(s_code_point[i])[0]
                    code[positions] = True
                    code[j] = True
                    actions.append(Action(code, positions[-1], 3))

    return actions


def get_legal_actions(actions: list[Action], previous_action: Action):
    legal_actions = [Action()]  # pass
    for action in actions:
        if action.count == previous_action.count:
            if action.type > previous_action.type:
                legal_actions.append(action)
            elif action.type == previous_action.type:
                if action.type == 1:  # flush
                    action_s_code = np.any(action.code.reshape(13, 4), axis=1)
                    pre_action_s_code = np.any(
                        previous_action.code.reshape(13, 4), axis=1
                    )
                    overlap = np.bitwise_and(action_s_code, pre_action_s_code)
                    action_s_code = np.bitwise_and(
                        action_s_code, np.bitwise_not(overlap)
                    )
                    pre_action_s_code = np.bitwise_and(
                        pre_action_s_code, np.bitwise_not(overlap)
                    )
                    if np.sum(action_s_code) != 0:  # having complete same number
                        max_action = get_max_index(action_s_code)
                        max_pre_action = get_max_index(pre_action_s_code)
                        if max_action > max_pre_action:
                            legal_actions.append(action)

                elif action.critical_card > previous_action.critical_card:
                    legal_actions.append(action)

    return legal_actions


def get_max_index(arr):
    return np.where(arr)[0][-1]


pair_matrix = np.array(
    [
        [True, True, False, False],
        [True, False, True, False],
        [True, False, False, True],
        [False, True, True, False],
        [False, True, False, True],
        [False, False, True, True],
    ]
)

triplet_matrix = np.array(
    [
        [True, True, True, False],
        [True, True, False, True],
        [True, False, True, True],
        [False, True, True, True],
    ]
)

stright_matrix = np.array(
    [
        [
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        [
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        [
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
        ],
        [
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
        ],
        [
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
            False,
        ],
        [
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
            False,
        ],
        [
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
            False,
        ],
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
            False,
        ],
        [
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            True,
            True,
            True,
            True,
            True,
        ],
    ]
)
