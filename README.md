# Estimation of Human-perceived Difficulty of Chess Positions

Environment:

```bash
conda create -n chess-ai python=3.11 numpy
conda activate chess-ai
pip install chess
conda install ipykernel
python -m ipykernel install --user --name chess-ai --display-name "chess-ai"
conda install matplotlib
conda install conda-forge::scipy
conda install conda-forge::scikit-learn
conda install conda-forge::pandas
conda install conda-forge::xgboost
conda install conda-forge::shap
```

Download stockfish from [here](https://stockfishchess.org/download/) and put it in C:/stockfish (no need to actually install the .exe). 

Dataset downloaded from [here](https://database.lichess.org/#puzzles). 
