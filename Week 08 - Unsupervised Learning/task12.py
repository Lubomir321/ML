import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import ast

def main():
    with open('price_movements.txt', 'r') as file:
        price_movements = ast.literal_eval(file.read())

    price_movements = np.array(price_movements)

    companies = [
        'Apple', 'AIG', 'Amazon', 'American express', 'Boeing', 'Bank of America', 
        'British American Tobacco', 'Canon', 'Caterpillar', 'Colgate-Palmolive', 'ConocoPhillips', 
        'Cisco', 'Chevron', 'DuPont de Nemours', 'Dell', 'Ford', 'General Electrics', 'Google/Alphabet', 
        'Goldman Sachs', 'GlaxoSmithKline', 'Home Depot', 'Honda', 'HP', 'IBM', 'Intel', 
        'Johnson & Johnson', 'JPMorgan Chase', 'Kimberly-Clark', 'Coca Cola', 'Lookheed Martin', 
        'MasterCard', 'McDonalds', '3M', 'Microsoft', 'Mitsubishi', 'Navistar', 'Northrop Grumman', 
        'Novartis', 'Pepsi', 'Pfizer', 'Procter Gamble', 'Philip Morris', 'Royal Dutch Shell', 'SAP', 
        'Schlumberger', 'Sony', 'Sanofi-Aventis', 'Symantec', 'Toyota', 'Total', 
        'Taiwan Semiconductor Manufacturing', 'Texas instruments', 'Unilever', 'Valero Energy', 
        'Walgreen', 'Wells Fargo', 'Wal-Mart', 'Exxon', 'Xerox', 'Yahoo'
    ]

    print(price_movements)
    normalized_movements = normalize(price_movements)
    print(normalized_movements)

    learning_rates = [10, 50, 71]

    for rate in learning_rates:
        tsne = TSNE(learning_rate=rate)
        tsne_features = tsne.fit_transform(normalized_movements)

        x = tsne_features[:, 0]
        y = tsne_features[:, 1]

        plt.figure(figsize=(12, 8))
        plt.scatter(x, y)

        for i, company in enumerate(companies):
            plt.annotate(company, (x[i], y[i]), fontsize=8, alpha=0.75)

        plt.title(f't-SNE on the stock price dataset')
        plt.tight_layout()
        plt.show()



if __name__ == '__main__':
    main()