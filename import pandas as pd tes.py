import pandas as pd
import numpy as np
import requests
from sklearn import tree
from sklearn.tree import export_graphviz
import graphviz
from io import StringIO

# URL tempat file CSV berada
csv_url = "https://github.com/ahmadkahfi98/testing/blob/main/testing.csv"

# Mengunduh data dari URL
response = requests.get(csv_url)

# Memeriksa apakah pengunduhan berhasil
if response.status_code == 200:
    # Membaca data CSV
    data = pd.read_csv(StringIO(response.text))

    # Menampilkan nama atribut
    print(data.columns)

    # Menampilkan data
    print(data.head())

    # Data yang akan dihapus
    removed = [0, 1, 2]  # Ganti indeks data yang ingin dihapus sesuai dengan data Anda

    # Menghapus data yang dihapus dari target
    new_target = np.delete(data['Hipertensi'].values, removed)  # Sesuaikan dengan kolom yang sesuai

    # Menghapus data yang dihapus dari atribut
    new_data = data.drop(data.index[removed])

    # Membuat dan melatih classifier Decision Tree dengan kriteria entropi
    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf = clf.fit(new_data.drop(columns=['Hipertensi']), new_target)  # Sesuaikan dengan kolom yang sesuai

    # Memasukkan data yang dihapus sebagai input untuk prediksi
    predictions = clf.predict(data.iloc[removed, 1:-1])

    # Menampilkan hasil prediksi
    for i in range(len(removed)):
        print(f"Data ke-{removed[i]} adalah '{predictions[i]}'")

    # Visualisasi pohon keputusan dalam format .dot
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=new_data.columns[1:-1],  
                                    class_names=data['Hipertensi'].unique(),  # Sesuaikan dengan kolom yang sesuai
                                    filled=True, rounded=True, special_characters=True)  
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree")  # Simpan visualisasi dalam bentuk file

    # Menampilkan visualisasi pohon entropi
    graph.view("decision_tree")
else:
    print("Gagal mengunduh data dari URL.")
