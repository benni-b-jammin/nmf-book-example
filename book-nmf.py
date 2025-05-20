#!/usr/bin/env python3
'''
book-nmf.py   - simple program that utilizes sklearn's NMF package to organize
                a book dataset and create a simple recommender. This program
                follows the tutorial created by Quin Daly (see README for 
                links to data and article)
                
Author:         Benji Lawrence
Last Modified:  May 20, 2025
'''
# necessary packages for implementation
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
import os

def main():
    # import data from Kaggle - link in README
    users_master = pd.read_csv('./book-data/Users.csv', sep=';')
    books_master = pd.read_csv('./book-data/Books.csv', sep=';')
    ratings_master = pd.read_csv('./book-data/Ratings.csv', sep=';')
    
    # formatting data - keep books with >= 50 ratings, users with >= 10 ratings
    ratings = ratings_master.copy()
    book_rating_group = ratings.groupby(['ISBN']).count()
    book_rating_group = book_rating_group[book_rating_group['Rating']>50]
    ratings = ratings[ratings['ISBN'].isin(book_rating_group.index)]
    user_rating_group = ratings.groupby(['User-ID']).count()
    user_rating_group = user_rating_group[user_rating_group['Rating']>10]
    ratings = ratings[ratings['User-ID'].isin(user_rating_group.index)]
        
    # apply to books and users datasets
    books = books_master.copy()
    books = books[books['ISBN'].isin(ratings['ISBN'])]
    users = users_master.copy()
    users = users[users['User-ID'].isin(ratings['User-ID'])]
        
    # load matrix if saved to file - else create new and save
    if (os.path.exists("./v_matrix.pkl")):
        v_matrix = pd.read_pickle("./v_matrix.pkl")
        original_v_matrix = pd.read_pickle("./original_v_matrix.pkl")
    else:
        # create matrix V with book ID rows, user ID cols
        ''' # below caused PerformanceWarning - attempt with pivot instead
        user_ids = ratings['User-ID'].unique().astype(str) # extract ids
        cols = np.concatenate((['ISBN'],user_ids))
        v_matrix = pd.DataFrame(columns=cols)
        book_ids = books['ISBN']
        v_matrix['ISBN'] = book_ids
        v_matrix['ISBN'] = v_matrix['ISBN'].astype(str)
    
        # populate dataframe with ratings
        for i in range(ratings.shape[0]):
            user_id = ratings['User-ID'].iloc[i]
            book_id = ratings['ISBN'].iloc[i]
            rating = ratings['Rating'].iloc[i]
            row = v_matrix[v_matrix['ISBN']==book_id].index
            if len(row)>0:
                row = row[0]
                v_matrix.loc[row, user_id] = rating
    
        v_matrix.columns = v_matrix.columns.astype(str)
        '''
        # using pivot
        v_matrix = ratings.pivot(index='ISBN', columns='User-ID', values='Rating')
    
        # original v_matrix used later for verification
        original_v_matrix = v_matrix.copy()
        # original_v_matrix = original_v_matrix.set_index('ISBN')
        original_v_matrix.fillna('No Ranking', inplace=True)
    
        # NaN's replaced with zero - books with no rating
        v_matrix.fillna(0,inplace=True)
        # v_matrix = v_matrix.set_index('ISBN')

        # save for future use
        v_matrix.to_pickle("./v_matrix.pkl")
        original_v_matrix.to_pickle("./original_v_matrix.pkl")
    
    print(v_matrix)
    optimal = 500 #hard coded for now
    # optimal = calculate_rank(v_matrix.values)[0]
    if (os.path.exists("./V.pkl")):
        V = pd.read_pickle("./V.pkl")
        W = pd.read_pickle("./W.pkl")
        H = pd.read_pickle("./H.pkl")
    else:
        # decompose using sklearn NMF
        model = NMF(n_components=optimal, init='random', random_state=0)
        w_matrix = model.fit_transform(v_matrix.values)
        h_matrix = model.components_
        
        # reconstruct new V matrix
        W = pd.DataFrame(w_matrix, index=v_matrix.index)  # Rows = ISBNs
        H = pd.DataFrame(h_matrix, columns=v_matrix.columns)  # Columns = User-IDs
        V = pd.DataFrame(np.dot(w_matrix, h_matrix), index=v_matrix.index, columns=v_matrix.columns)
        
        # save for easier access
        V.to_pickle("./V.pkl")
        W.to_pickle("./W.pkl")
        H.to_pickle("./H.pkl")

    print(V)
    test_recommend(V, v_matrix, books)
    


def calculate_rank(data):
    '''
    Calculate optimal rank of dataframe
    '''
    matrix = data
    benchmark = np.linalg.norm(matrix, ord='fro') * 0.0001
    
    rank = 800 # initialize higher to improve speed
    while True:
        
        # initialize the model
        model = NMF(n_components=rank, init='random', random_state=0, max_iter=500)
        W = model.fit_transform(matrix)
        H = model.components_
        V = W @ H
        
        # Calculate RMSE of original df and new V
        RMSE = np.sqrt(mean_squared_error(matrix, V))
        
        if RMSE < benchmark:
            return rank, V
        
        # Increment rank if RMSE isn't smaller than the benchmark
        print(f"rank={rank}, RMSE={RMSE}, benchmark={benchmark}")
        rank += 1

    return rank

def test_recommend(V, original_df, books, user_id = 278633):
    # Grab top 10 books ID's that the user hasn't reviewed
    user_col = V[user_id]
    user_col = user_col.sort_values(ascending=False)
    
    top_10_ISBN = []
    for book in user_col.index:
        if original_df[user_id].loc[book] == 0: # haven't read the book
            top_10_ISBN.append(book)
    
        if len(top_10_ISBN) == 10:
            break
    
    print(top_10_ISBN)

    # Return the titles and authors of the recommended books
    books_df = books.set_index('ISBN')
    books_df.index = books_df.index.astype(str)
    
    top_10_books = []
    for book in top_10_ISBN:
        top_10_books.append([books_df['Title'].loc[book], books_df['Author'].loc[book]])
    
    top_10_books = pd.DataFrame(top_10_books, columns=['Title','Author'])
    
    print(f'Top 10 Book Recommendations for user {user_id}:')
    print(top_10_books)
    


if __name__ == "__main__":
    main()
