from stockfish import Stockfish
import chess
from tqdm import tqdm

def read_n_lines(file, n, skip=0):
    f = open(file)
    out = []
    f.readline()
    for _ in range(skip):
        if not f.readline():
            return out

    for _ in range(n):
        line = f.readline()
        if not line:
            break
        out.append(line)

    f.close()
    return out

def get_stockfish_attributes(stockfish, fen, to_move):
    out = {}
    stockfish.set_fen_position(fen)
    eval = stockfish.get_evaluation()
    if eval["type"] == "cp":
        out["cp_eval"] = eval["value"]
    else:
        if to_move == "w":
            out["cp_eval"] = 15 * 100
        else:
            out["cp_eval"] = -15 * 100

    top_moves = stockfish.get_top_moves(3, verbose=True)
    for i, top in enumerate(top_moves):
        if top["Mate"]:
            if to_move == "w":
                out[f"move{i+1}cp"] = 15 * 100
            else:
                out[f"move{i+1}cp"] = -15 * 100
        else:
            out[f"move{i+1}cp"] = top["Centipawn"]
        out["nodes"] = top["Nodes"]
        out[f"move{i+1}multiPV"] = top["MultiPVNumber"]
        out[f"move{i+1}sel_depth"] = top["SelectiveDepth"]
        w, d, l = top["WDL"].split(" ")
        out[f"move{i+1}w"] = w
        out[f"move{i+1}d"] = d
        out[f"move{i+1}l"] = l
    w, d, l = stockfish.get_wdl_stats()
    out["orig_w"] = w
    out["orig_d"] = d
    out["orig_l"] = l
    return out

def write_to_file(file, data, nl=False):
    f = open(file, "a+")
    if nl:
        f.write("\n")
    for d in data:
        f.write(str(d)+"\n")
    f.close()

def construct_dataset(in_file, out_file, n, skip=0):
    stockfish = Stockfish(path="./stockfish-windows-x86-64-avx2.exe")
    count = 0
    lines = read_n_lines(in_file, n, skip)
    dataset = []
    for line in tqdm(lines):
        curr = {}
        id,fen,moves,rating,ratingdev,pop,nbplays,themes,gameurl,openingtags = line.split(",")
        firstmove = chess.Move.from_uci(moves.split(" ")[0])
        b = chess.Board(fen)
        b.push(firstmove)
        fen = b.fen()
        epd = b.epd()

        curr["themes"] = themes
        curr["solution"] = moves
        curr["epd"] = epd
        curr["rating"] = rating
        curr["rating_dev"] = ratingdev
        curr["to_move"] = epd.split(" ")[-3]

        sf_atts = get_stockfish_attributes(stockfish, fen, curr["to_move"])
        for k, v in sf_atts.items():
            curr[k] = v
        dataset.append(curr)
        #print(f"Progress: {count}/{n}       ", end="\r", flush=True)
        count += 1
    write_to_file(out_file, dataset)

if __name__ == "__main__":
    n = 100000
    in_file = "./lichess_db_puzzle.csv"
    out_file = "./dataset_test.txt"
    skip = 0
    construct_dataset(in_file, out_file, n, skip)