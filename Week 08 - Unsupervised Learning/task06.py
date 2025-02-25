import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import Pipeline
import ast

def main():
    with open('price_movements.txt', 'r') as file:
        price_movements = ast.literal_eval(file.read())

    price_movements = np.array(price_movements)

    print(f"Data shape: {price_movements.shape}")

    companies = ['Apple', 'AIG', 'Amazon', 'American express', 'Boeing', 'Bank of America', 'British American Tobacco', 'Canon', 'Caterpillar', 'Colgate-Palmolive', 'ConocoPhillips', 'Cisco', 'Chevron', 'DuPont de Nemours', 'Dell', 'Ford', 'General Electrics', 'Google/Alphabet', 'Goldman Sachs', 'GlaxoSmithKline', 'Home Depot', 'Honda', 'HP', 'IBM', 'Intel', 'Johnson & Johnson', 'JPMorgan Chase', 'Kimberly-Clark', 'Coca Cola', 'Lookheed Martin', 'MasterCard', 'McDonalds', '3M', 'Microsoft', 'Mitsubishi', 'Navistar', 'Northrop Grumman', 'Novartis', 'Pepsi', 'Pfizer', 'Procter Gamble', 'Philip Morris', 'Royal Dutch Shell', 'SAP', 'Schlumberger', 'Sony', 'Sanofi-Aventis', 'Symantec', 'Toyota', 'Total', 'Taiwan Semiconductor Manufacturing', 'Texas instruments', 'Unilever', 'Valero Energy', 'Walgreen', 'Wells Fargo', 'Wal-Mart', 'Exxon', 'Xerox', 'Yahoo']

    pipeline = Pipeline([
        ('normalizer', Normalizer()),
        ('kmeans', KMeans(n_clusters=10, random_state=42))
    ])

    pipeline.fit(price_movements)

    labels = pipeline.named_steps['kmeans'].labels_

    clustered_companies = pd.DataFrame({
        'labels': labels,
        'companies': companies
    })

    clustered_companies = clustered_companies.sort_values('labels')
    print(clustered_companies)



if __name__ == '__main__':
    main()