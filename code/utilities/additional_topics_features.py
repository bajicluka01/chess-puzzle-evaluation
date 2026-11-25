import chess

PIECE_VALUES_PINS = {
    "P": 1, "p": 1,
    "N": 3, "n": 3,
    "B": 3, "b": 3,
    "R": 5, "r": 5,
    "Q": 9, "q": 9,
    "K": 100, "k": 100
}

def check_pawn_attack(board, pawn, pawn_square, attacker_piece, attacker_square):
    """
        Function that returns 1 the given epd is a fork/pin or not for a special cases.

        In case of fork:
            If two pieces are attacked at the same time and one of them is pawn, we consider this as a special case. 
            If the pawn is not defended, we do consider this as a fork. 
            If the pawn is defended, in general we do not consider this as a fork, except if these two conditions hold
                - when a pawn is defended by a king and only king
                - this pawn is also attacked by some other piece than the one that did the fork

        Similar logic applies to a case of a pin where we have something like piece -> opponent_piece -> opponent_pawn 
        (However, not piece -> opponent_pawn -> opponent_piece. This is automatically considered a pin)
    """

    # 1. if the pawn is not defended, return 1
    defenders_of_pawn_board = board.attackers(pawn.color, pawn_square)
    defenders_of_pawn = [chess.square_name(sq) for sq in defenders_of_pawn_board]

    if len(defenders_of_pawn) == 0:
        return 1

    # 2. if it is defended by 2 pieces or more, return 0
    if len(defenders_of_pawn) > 1:
        return 0

    # 3a. now we check if its only defending is a king
    if board.piece_at(chess.parse_square(defenders_of_pawn[0])).piece_type == chess.KING:
        # 3b. now we check whether this pawn is attacked by one more piece (that is not attacker piece)
        other_attackers_board = board.attackers(not pawn.color, pawn_square)
        #other_attackers = [chess.square_name(sq) for sq in other_attackers_board]
        if any(sq != attacker_square for sq in other_attackers_board):
            return 1
        else:
            return 0

    # 4. if the only defending piece is not king, return 0
    return 0

def find_specific_pin(epd, pinning_piece):
    """
        Finds and returns all pins of a specific piece (queen, rook or bishop) on the current board
    """

    board = chess.Board()
    board.set_epd(epd)

    pins = []
    for piece_square, piece in board.piece_map().items():
        # among all pieces find the pinning piece
        if piece.piece_type != pinning_piece:
            continue

        # go through the squares this piece is attacking
        for attacked_square in board.attacks(piece_square):
            # find a square that has opponents piece on it
            attacked_piece = board.piece_at(attacked_square)
            if attacked_piece is None or attacked_piece.color == piece.color:
                continue

            # now we need to check if there is any piece behind it

            # we find in which direction we will be increasing or decreasing file and row
            file_direction = chess.square_file(attacked_square) - chess.square_file(piece_square)
            if file_direction != 0:     # if it's on the same file
                file_direction = int(file_direction / abs(file_direction))
            row_direction = chess.square_rank(attacked_square) - chess.square_rank(piece_square)
            if row_direction != 0:      # if it's on the same row
                row_direction = int(row_direction / abs(row_direction))

            direction = (file_direction, row_direction)

            file, row = (
                chess.square_file(attacked_square) + direction[0],
                chess.square_rank(attacked_square) + direction[1],
            )

            # now we check until we find a piece or until we come to the end of the board
            while 0 <= file < 8 and 0 <= row < 8:
                current_square = chess.square(file, row)
                current_piece = board.piece_at(current_square)

                # until we find a piece
                if current_piece is not None:

                    # check if it's opponent piece or a pawn
                    if current_piece.color == piece.color:
                        break

                    # if it's a pawn it's a special case 
                    if current_piece.piece_type == chess.PAWN:

                        its_a_pin = check_pawn_attack(board, pawn=current_piece, pawn_square=current_square, attacker_piece=piece, attacker_square=piece_square)
                        if its_a_pin:
                            pins.append({
                                "pinning_piece": piece.symbol(),
                                "pinning_piece_square": chess.square_name(piece_square),
                                "pinned_piece": attacked_piece.symbol(),
                                "pinned_piece_square": chess.square_name(attacked_square),
                                "piece_behind": current_piece.symbol(),
                                "piece_behind_square": chess.square_name(current_square),
                            })

                        break

                    # we check whether the piece behind is defended or not 
                    # also, it doesn't count if it's defended by the pinned piece
                    defenders_board = board.attackers(current_piece.color, current_square)
                    defenders = [chess.square_name(sq) for sq in defenders_board]

                    # exclude pinned piece from defenders
                    defenders = [sq for sq in defenders if sq != chess.square_name(attacked_square)]

                    # if there's no other defender, this is for sure a pin
                    is_defended = len(defenders) > 0
 
                    # now the current piece either has higher value than the pinned one or is undefendend
                    if (
                        PIECE_VALUES_PINS[current_piece.symbol()] > PIECE_VALUES_PINS[piece.symbol()]
                        or not
                        is_defended
                    ):
                        pins.append({
                            "pinning_piece": piece.symbol(),
                            "pinning_piece_square": chess.square_name(piece_square),
                            "pinned_piece": attacked_piece.symbol(),
                            "pinned_piece_square": chess.square_name(attacked_square),
                            "piece_behind": current_piece.symbol(),
                            "piece_behind_square": chess.square_name(current_square),
                        })

                    break

                file += direction[0]
                row += direction[1]
    return pins

def find_all_pins(epd):
    """
        Function that returns all pins (created by queen, rook and bishop) from given epd.

        Returns it as a list corresponding to dataset:
            [bishop_pin_1, ... , bishop_pin_p, rook_pin_1, ... , rook_pin_q, queen_pin_1, ... , queen_pin_r,]

        Each pin is a dictionary describing a piece that is attacking (queen, rook or bishop),
        which piece it's attacking, and which piece is behind it. For example:
            {
                "pinning_piece": R,
                "pinning_piece_square": "a1",
                "pinned_piece": "n",
                "pinned_piece_square": "a5",
                "piece_behind": "q",
                "piece_behind_square": "a8",
            }
    """

    """
        General definition of a pin:
            a piece attacks a piece of the opponent
            -> if the attacked piece has a higher valued piece of itself in the back, it is considered pinned
            -> if the attacked piece has equal or lower valued piece of itself in the back, this piece needs to
            be undefeneded for attacked piece to be considered pinned
    """

    pins = []

    pins.extend(find_specific_pin(epd, chess.BISHOP))
    pins.extend(find_specific_pin(epd, chess.ROOK))
    pins.extend(find_specific_pin(epd, chess.QUEEN))

    return pins

def find_all_forks_threatening_knight(epd):
    """
        Function that returns all potential forks (created by a knight) from given epd.
    """

    board = chess.Board()
    board.set_epd(epd)

    forks = []

    for knight_square, knight in board.piece_map().items():
        # if the piece is not a knight, continue
        if knight.piece_type != chess.KNIGHT:
            continue

        # go through all possible jumps of this knight
        for jump_square in board.attacks(knight_square):
            # check whether this square is occupied by their own piece
            jump_square_occupation = board.piece_at(jump_square)
            if jump_square_occupation is not None and jump_square_occupation.color == knight.color:
                continue

            # find the opposite pieces that can be attacked by a knight from this square
            forked_pieces = []
            for attacked_square in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[jump_square]):
                potentially_attacked_piece = board.piece_at(attacked_square)

                # if there is a piece and it's opponents piece, store it
                if potentially_attacked_piece is not None and potentially_attacked_piece.color != knight.color:
                    forked_pieces.append({
                        "piece": potentially_attacked_piece.symbol(),
                        "square": chess.square_name(attacked_square),
                    })

            # filter out pawns that do not count as forked pieces
            filtered_forked_pieces = []

            for forked_piece in forked_pieces:
                if forked_piece["piece"].lower() == "p":
                    pawn_square = chess.parse_square(forked_piece["square"])
                    pawn = board.piece_at(pawn_square)

                    its_a_fork = check_pawn_attack(board, pawn=pawn, pawn_square=pawn_square, attacker_piece=knight, attacker_square=jump_square)

                    if its_a_fork:
                        filtered_forked_pieces.append(forked_piece)
                else:
                    filtered_forked_pieces.append(forked_piece)

            forked_pieces = filtered_forked_pieces

            # if there is more than one piece that can be forked, we store this as fork
            if len(forked_pieces) >= 2:
                forks.append({
                    "piece": knight.symbol(),
                    "piece_square": chess.square_name(knight_square),
                    "piece_to": chess.square_name(jump_square),
                    "forked_pieces": forked_pieces,
                })
    return forks

def find_all_forks_threatening_pawn(epd):
    """
        Function that returns all potential forks (created by a pawn) from given epd.
    """

    board = chess.Board()
    board.set_epd(epd)

    forks = []

    for pawn_square, pawn in board.piece_map().items():
        # if the piece is not a pawn, continue
        if pawn.piece_type != chess.PAWN:
            continue

        # pawns move towards rank 8 if white, rank 1 if black
        if pawn.color == chess.WHITE:
            direction = 1
            starting_rank = 1
        else:
            direction = -1 
            starting_rank = 6

        file = chess.square_file(pawn_square)
        row = chess.square_rank(pawn_square)
        possible_ranks = [row + direction]

        # check if its on its starting point, then it can move for two as well
        if row == starting_rank:
            possible_ranks.append(row + 2 * direction)

        # go through all possible moves of this pawn
        for target_rank in possible_ranks:
            # safe net
            if not 0 <= target_rank <= 7:
                continue

            jump_square = chess.square(file, target_rank)

            # check whether this square is occupied by any piece
            jump_square_occupation = board.piece_at(jump_square)
            if jump_square_occupation is not None:
                continue

            # find the opposite pieces that can be attacked diagonally by a pawn from this square
            forked_pieces = []
            for attacked_file in [file - 1, file + 1]:
                # if on edge skip
                if not 0 <= attacked_file <= 7:
                    continue

                attacked_square = chess.square(attacked_file, target_rank + direction)

                if not 0 <= chess.square_rank(attacked_square) <= 7:
                    continue

                # if there is a piece and it's opponents piece, store it
                potentially_attacked_piece = board.piece_at(attacked_square)

                if potentially_attacked_piece is not None and potentially_attacked_piece.color != pawn.color:
                    forked_pieces.append({
                        "piece": potentially_attacked_piece.symbol(),
                        "square": chess.square_name(attacked_square),
                    })

            """
            #print(f"forked pieces = {forked_pieces}")
            # filter out pawns that do not count as forked pieces
            filtered_forked_pieces = []

            for forked_piece in forked_pieces:
                if forked_piece["piece"].lower() == "p":
                    pawn_square_target = chess.parse_square(forked_piece["square"])
                    pawn_target = board.piece_at(pawn_square_target)

                    its_a_fork = check_pawn_attack(board, pawn=pawn_target, pawn_square=pawn_square_target, attacker_piece=pawn, attacker_square=jump_square)

                    if its_a_fork:
                        filtered_forked_pieces.append(forked_piece)
                else:
                    filtered_forked_pieces.append(forked_piece)

            #print(f"filtered forked pieces = {filtered_forked_pieces}")
            

            forked_pieces = filtered_forked_pieces
            """

            # if there is more than one piece that can be forked, we store this as fork
            if len(forked_pieces) >= 2:
                forks.append({
                    "piece": pawn.symbol(),
                    "piece_square": chess.square_name(pawn_square),
                    "piece_to": chess.square_name(jump_square),
                    "forked_pieces": forked_pieces,
                })
    return forks

def find_all_forks_threatening_brq(epd, wanted_piece):
    """
        Function that returns all potential forks (created by a bishop, rook, or a queen) from given epd.
    """

    board = chess.Board()
    board.set_epd(epd)

    forks = []

    for piece_square, piece in board.piece_map().items():
        # if the piece is not a requested piece, continue
        if piece.piece_type != wanted_piece:
            continue

        # go through the squares this piece attacks
        for target_square in board.attacks(piece_square):
            target_occupation = board.piece_at(target_square)

            # cannot move onto a square occupied by our own piece
            if (
                target_occupation is not None
                and target_occupation.color == piece.color
            ):
                continue

            # make a temporary board and simulate the move
            temp_board = board.copy()
            temp_board.remove_piece_at(piece_square)
            temp_board.remove_piece_at(target_square)
            temp_board.set_piece_at(target_square, piece)

            # find opponent pieces attacked from the new square
            forked_pieces = []

            for attacked_square in temp_board.attacks(target_square):
                attacked_piece = temp_board.piece_at(attacked_square)

                if (
                    attacked_piece is not None
                    and attacked_piece.color != piece.color
                ):
                    forked_pieces.append({
                        "piece": attacked_piece.symbol(),
                        "square": chess.square_name(attacked_square),
                    })

            # filter out pawns that do not count as forked pieces
            filtered_forked_pieces = []

            for forked_piece in forked_pieces:
                if forked_piece["piece"].lower() == "p":
                    pawn_square = chess.parse_square(forked_piece["square"])
                    pawn = temp_board.piece_at(pawn_square)

                    its_a_fork = check_pawn_attack(temp_board, pawn=pawn, pawn_square=pawn_square, attacker_piece=piece, attacker_square=target_square)

                    if its_a_fork:
                        filtered_forked_pieces.append(forked_piece)
                else:
                    filtered_forked_pieces.append(forked_piece)

            forked_pieces = filtered_forked_pieces

            # store the fork if at least two opponent pieces are attacked
            if len(forked_pieces) >= 2:
                forks.append({
                    "piece": piece.symbol(),
                    "piece_square": chess.square_name(piece_square),
                    "piece_to": chess.square_name(target_square),
                    "forked_pieces": forked_pieces,
                })
    return forks

def find_all_forks_threatening(epd):
    """
        Returns it as a list of forks:
            [fork_1, fork_2, ... , fork_n]

        Each fork is a dictionary describing a knight move and the opponent pieces
        that would be attacked by the knight after making that move. For example:
            {
                "piece": "N",
                "piece_square": "c3",
                "piece_to": "d5",
                "forked_pieces": [
                    {
                        "piece": "q",
                        "square": "e7",
                    },
                    {
                        "piece": "r",
                        "square": "f6",
                    }
                ],
            }
    """

    """
        A potential fork is defined as a knight move to a square from which the
        knight would attack at least two opponent pieces. It does not matter whether
        the piece is defended or not, it also takes into account pawns. The latter 
        is a special case described in check_pawn_attack(). It returns
        forks from both sides.
    """

    pawn_forks = find_all_forks_threatening_pawn(epd)
    knight_forks = find_all_forks_threatening_knight(epd)
    bishop_forks = find_all_forks_threatening_brq(epd, chess.BISHOP)
    rook_forks = find_all_forks_threatening_brq(epd, chess.ROOK)
    queen_forks = find_all_forks_threatening_brq(epd, chess.QUEEN)
    
    all_forks = pawn_forks + knight_forks + bishop_forks + rook_forks + queen_forks
    return all_forks

def find_all_forks_current_specific_piece(epd, wanted_piece):
    """
        Function that returns all forks from given epd.

        Returns it as a list of forks:
            [fork_1, fork_2, ... , fork_n]

        Each fork is a dictionary describing the piece and the opponent pieces
        currently attacked by that piece. For example:
            {
                "piece": "N",
                "piece_square": "c5",
                "forked_pieces": [
                    {
                        "piece": "q",
                        "square": "e7",
                    },
                    {
                        "piece": "r",
                        "square": "f6",
                    }
                ],
            }
    """

    """
        A current fork is defined as a piece that is currently attacking at least
        two opponent pieces from its current square. It does not matter whether
        the piece is defended or not, it also takes into account pawns. The latter 
        is a special case described in check_pawn_attack(). It returns
        forks from both sides.
    """

    board = chess.Board()
    board.set_epd(epd)

    forks = []

    for piece_square, piece in board.piece_map().items():
        # if the piece is not a knight, continue
        if piece.piece_type != wanted_piece:
            continue

        # find the opposite pieces that can be attacked by a knight from this square
        forked_pieces = []
        for attacked_square in board.attacks(piece_square):
            potentially_attacked_piece = board.piece_at(attacked_square)

            # if there is a piece and it's opponents piece, store it
            if potentially_attacked_piece is not None and potentially_attacked_piece.color != piece.color:
                forked_pieces.append({
                    "piece": potentially_attacked_piece.symbol(),
                    "square": chess.square_name(attacked_square),
                })

        # filter out pawns that do not count as forked pieces
        filtered_forked_pieces = []

        for forked_piece in forked_pieces:
            if forked_piece["piece"].lower() == "p":
                pawn_square = chess.parse_square(forked_piece["square"])
                pawn = board.piece_at(pawn_square)

                its_a_fork = check_pawn_attack(board, pawn=pawn, pawn_square=pawn_square, attacker_piece=piece, attacker_square=piece_square)

                if its_a_fork:
                    filtered_forked_pieces.append(forked_piece)
            else:
                filtered_forked_pieces.append(forked_piece)

        forked_pieces = filtered_forked_pieces

        # if there is more than one piece that can be forked, we store this as fork
        if len(forked_pieces) >= 2:
            forks.append({
                "piece": piece.symbol(),
                "piece_square": chess.square_name(piece_square),
                "forked_pieces": forked_pieces,
            })
    return forks

def find_all_forks_current(epd):
    pawn_forks = find_all_forks_current_specific_piece(epd, chess.PAWN)
    knight_forks = find_all_forks_current_specific_piece(epd, chess.KNIGHT)
    bishop_forks = find_all_forks_current_specific_piece(epd, chess.BISHOP)
    rook_forks = find_all_forks_current_specific_piece(epd, chess.ROOK)
    queen_forks = find_all_forks_current_specific_piece(epd, chess.QUEEN)

    all_forks = pawn_forks + knight_forks + bishop_forks + rook_forks + queen_forks
    return all_forks