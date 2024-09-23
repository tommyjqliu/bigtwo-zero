import numpy as np
from itertools import combinations, product


def get_max_index(arr):
    return np.where(arr)[0][-1]


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
    # TODO: accelerate
    # Create an empty 52-length boolean array
    code_point = np.zeros(52, dtype=bool)
    for card in cards:
        # Extract rank and suit from the card string
        rank = card[0]
        suit = card[1]
        # Compute the index based on rank and suit order
        rank_index = ranks.index(rank)
        suit_index = suits.index(suit)
        # Calculate the final index in the 52-length array
        index = rank_index * 4 + suit_index
        # Set the corresponding position to True
        code_point[index] = True

    return code_point


class Action:
    def __init__(self, code=np.zeros(52, dtype=bool), critical_card=0, trick_type=0):
        self.count = np.sum(code)
        self.is_pass = self.count == 0
        self.code = code
        self.type = trick_type
        self.critical_card = critical_card


class Player:
    def __init__(self, holding):
        self.holding = holding
        self.played = np.zeros(52, dtype=bool)
        self.all_actions = possible_actions(holding)
        self.legal_actions: list[Action] = []
        self.history = []

    def update(self, action_code):
        t = np.bitwise_xor(self.holding, action_code)
        t = np.bitwise_and(action_code, t)
        assert np.sum(t) == 0 # check if the action is valid
        self.played = np.bitwise_or(self.played, action_code)
        self.holding = np.bitwise_xor(self.holding, action_code)
        next_all_actions = []

        for action in self.all_actions:
            if np.sum(np.bitwise_and(action.code, self.holding)) == action.count:
                next_all_actions.append(action)
        self.all_actions = next_all_actions

    def pre_action(self, action_in_charge: Action):
        self.legal_actions = get_legal_actions(self.all_actions, action_in_charge)


def possible_actions(code_point):
    s_code_point = code_point.reshape(13, 4)
    s_sum = s_code_point.sum(axis=1)
    r_code_point = code_point.reshape(13, 4).T
    r_sum = r_code_point.sum(axis=1)
    actions = []
    possible_2 = {}
    possible_3 = {}
    strightflush =  np.empty((0, 52), dtype=bool)

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


def get_action_type(action_code):
    if np.sum(action_code) < 5:
        return 0
    s_code_point = action_code.reshape(13, 4)
    s_sum = s_code_point.sum(axis=1)
    r_code_point = action_code.reshape(13, 4).T
    r_sum = r_code_point.sum(axis=1)

    is_stright = (
        np.sum(np.sum(np.bitwise_and(stright_matrix, s_sum != 0), axis=1) == 5) > 0
    )
    is_flush = np.sum(r_sum >= 5) > 0
    is_fullhouse = np.sum(s_sum >= 3) > 0
    is_fourkind = np.sum(s_sum >= 4) > 0
    is_strightflush = is_stright and is_flush

    return get_max_index(
        [is_stright, is_flush, is_fullhouse, is_fourkind, is_strightflush]
    )


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
                    if np.sum(action_s_code) != 0: # having complete same number
                        max_action = get_max_index(action_s_code)
                        max_pre_action = get_max_index(pre_action_s_code)
                        if max_action > max_pre_action:
                            legal_actions.append(action)

                elif action.critical_card > previous_action.critical_card:
                    legal_actions.append(action)

    return legal_actions


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
