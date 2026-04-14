import matplotlib.pyplot as plt
import seaborn as sns

def plot_rating_dist(file):
    f = open(file)
    i = 1
    out = []
    line = f.readline() # to ignore header
    while True:
        line = f.readline()
        if len(line) <= 1:
            break
        id,fen,moves,rating,ratingdev,pop,nbplays,themes,gameurl,openingtags = line.split(",")
        out.append(rating)
    f.close()
    
    sns.distplot(out, hist=True, kde=False, 
             bins=30, color = 'blue',
             hist_kws={'edgecolor':'black'})
    plt.title("Rating distribution")
    plt.show()

if __name__ == "__main__":
    plot_rating_dist("lichess_db_puzzle.csv")
