import argparse
import numpy as np
import pandas as pd
import csv
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Загружаем данные
products = pd.read_csv("new_sportsGoodsDescription.csv")


def check_rating_count(user_id):
    with open("new_sportsGoodsDescription.csv", encoding="utf-8") as file:
        target_user_id = user_id
        count = 0
        next(csv.reader(file))
        for line in csv.reader(file):
            if int(line[5]) == target_user_id:
                count += 1
        return count


def create_matrix(df):
    N = len(df["user_id"].unique())
    M = len(df["product_id"].unique())

    user_mapper = dict(zip(np.unique(df["user_id"]), list(range(N))))
    product_mapper = dict(zip(np.unique(df["product_id"]), list(range(M))))

    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["user_id"])))
    product_inv_mapper = dict(zip(list(range(M)), np.unique(df["product_id"])))

    user_index = [user_mapper[i] for i in df["user_id"]]
    product_index = [product_mapper[i] for i in df["product_id"]]

    X = csr_matrix((df["user_rating"], (product_index, user_index)), shape=(M, N))

    return X, user_mapper, product_mapper, user_inv_mapper, product_inv_mapper


X, user_mapper, product_mapper, user_inv_mapper, product_inv_mapper = create_matrix(products)


def find_similar(product_id, X, k, metric="cosine", show_distance=False):
    neighbour_list = []
    product_ind = product_mapper[product_id]
    product_vec = X[product_ind]
    k += 1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    product_vec = product_vec.reshape(1, -1)
    neighbour = kNN.kneighbors(product_vec, return_distance=show_distance)
    for i in range(0, k):
        n = neighbour.item(i)
        if n + 1 != product_ind:
            neighbour_list.append(product_inv_mapper[n])
    return neighbour_list


def recommend_product_for_user(user_id, X, user_mapper, product_mapper, product_inv_mapper, k=10):
    df1 = products[products["user_id"] == user_id]

    if df1.empty:
        return f"Пользователь с ID {user_id} не существует."

    product_id = df1[df1["user_rating"] == max(df1["user_rating"])]["product_id"].iloc[0]

    product_titles = dict(zip(products["product_id"], products["product_name"]))

    similar_ids = find_similar(product_id, X, k)
    product_title = product_titles.get(product_id, "Товар не найден")

    if product_title == "Товар не найден":
        return f"Товар с ID {product_id} не найден."

    recommendations = [product_titles.get(i, "Товар не найден") for i in similar_ids]
    return f"Поскольку вы оценили {product_title}, вам так же может понравится: {', '.join(recommendations)}"


def main():
    parser = argparse.ArgumentParser(description="Product recommendation system")
    parser.add_argument("user_id", type=int, help="ID пользователя для получения рекомендаций")
    args = parser.parse_args()

    user_id = args.user_id
    if user_id not in user_mapper:
        print("Некорректный ID пользователя.")
        return

    if check_rating_count(user_id) > 1:
        print(recommend_product_for_user(user_id, X, user_mapper, product_mapper, product_inv_mapper))
    else:
        print(f"Недостаточно данных для рекомендаций пользователю с ID {user_id}.")


if __name__ == "__main__":
    main()
