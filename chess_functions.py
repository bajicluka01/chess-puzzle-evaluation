import chess 
import chess.syzygy
import chess.svg
import networkx as nx
import os
import numpy as np
import requests
from bs4 import BeautifulSoup
import chess.engine
import matplotlib.pyplot as plt
from stockfish import Stockfish
import warnings
warnings.filterwarnings("ignore")

def download_syzygy_files(output_dir="syzygy_345"):
    # Configuration
    url = "http://tablebase.sesse.net/syzygy/3-4-5/"

    # This header tells the server you are a browser, not a bot
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Connecting to {url}...")
    try:
        # Pass the HEADERS here
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    links = [urljoin(url, a['href']) for a in soup.find_all('a', href=True) 
             if a['href'].endswith(('.rtbw', '.rtbz'))]

    print(f"Found {len(links)} files. starting download...")

    for i, file_url in enumerate(links, 1):
        filename = os.path.basename(file_url)
        file_path = os.path.join(output_dir, filename)

        if os.path.exists(file_path):
            print(f"[{i}/{len(links)}] Skipping {filename}")
            continue

        try:
            # Pass HEADERS here too for the actual file download
            with requests.get(file_url, headers=HEADERS, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=16384):
                        f.write(chunk)
            print(f"[{i}/{len(links)}] Finished: {filename}")
        except Exception as e:
            print(f"[{i}/{len(links)}] Error downloading {filename}: {e}")

def plot(board, arrows=[]):
    return chess.svg.board(board, arrows=arrows, size=500)  

def make_stockfish_moves(stockfish, moves):
    stockfish.make_moves_from_current_position(moves)

if __name__ == "__main__":
    # download tablebase
    #download_syzygy_files()

    board = chess.Board("8/1k6/8/2n3R1/8/4K3/8/8 w - - 0 1")
    tablebase = chess.syzygy.open_tablebase("C:/syzygy")

    stockfish = Stockfish(path="C:/stockfish/stockfish-windows-x86-64-avx2")
    #stockfish.set_elo_rating(3190)

    stockfish.set_fen_position(board.fen())
    best = stockfish.get_best_move()
    print(best)
    print(stockfish.get_evaluation())
    print(stockfish.get_static_eval())

    n_moves = 3
    candidate_moves = stockfish.get_top_moves(n_moves, verbose=True)
    print(candidate_moves[0])

    #print(stockfish.is_fen_valid(board.fen()))

    moves = ["e3d4", "c5e6"]
    make_stockfish_moves(stockfish, moves)
    print(stockfish.get_evaluation())
    print(stockfish.get_static_eval())

    new_board = chess.Board(stockfish.get_fen_position())

    print(stockfish.get_board_visual())

    print(stockfish.is_move_legal('a2a3'))

    # The perft command is used to test the move generation. It counts the total number of leaf nodes to a 
    # certain depth, and shows how this node count is divided amongst all legal moves of the current position.
    # The depth parameter should be an integer greater than zero and specifies the search depth.
    perft = stockfish.get_perft(3)
    print(perft)
    
    wdl = stockfish.get_wdl_stats()
    print(wdl)

    print(stockfish.info())
    print(stockfish.get_engine_parameters())

