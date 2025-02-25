import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
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

    normalized_movements = normalize(price_movements)

    mergings = linkage(normalized_movements, method='complete')

    plt.figure(figsize=(15, 10))
    dendrogram(mergings, labels=companies, leaf_rotation=90, leaf_font_size=6)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()