# Chess-endgame-network-analysis

Environment:

```bash
conda create -n chess-ai python=3.11 numpy
conda activate chess-ai
pip install chess
conda install ipykernel
python -m ipykernel install --user --name chess-ai --display-name "chess-ai"
conda install requests beautifulsoup4 (?)
conda install networkx (?)
conda install ipywidgets (?)
conda install matplotlib
conda install conda-forge::scipy
conda install conda-forge::scikit-learn
conda install conda-forge::pandas
conda install conda-forge::xgboost
conda install conda-forge::shap
conda install pygraphviz (?)
```

Download stockfish from [here](https://stockfishchess.org/download/) and put it in C:/stockfish (no need to actually install the .exe). 

Also put syzygy in C:/syzygy (code for download is in chess_functions.py). 

Dataset downloaded from [here](https://database.lichess.org/#puzzles) and processed with the function `construct_dataset` in `chess_functions.py`.
