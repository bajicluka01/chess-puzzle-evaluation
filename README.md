# Chess-endgame-network-analysis

Environment:

```bash
conda create -n chess-ao python=3.11 numpy
conda activate chess-ao
pip install chess
conda install ipykernel
python -m ipykernel install --user --name myenv --display-name "chess-ao"
conda install requests beautifulsoup4
conda install networkx
conda install ipywidgets
conda install matplotlib
conda install anaconda::scipy
conda install pygraphviz
```

Download stockfish from [here](https://stockfishchess.org/download/) and put it in C:/stockfish. Also put syzygy in C:/syzygy (code for download is in chess_functions.py. 
