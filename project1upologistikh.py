"""
Dataset φοιτητών:
https://archive.ics.uci.edu/dataset/320/student+performance
Χρησιμοποίησα pandas, matplotlib, sklearn,K-Means Clustering,KNN(Nearest Neighbors)
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# Φορτώνει δεδομένα φοιτητών
students = pd.read_csv("student-mat.csv", sep=";")

# Κρατάω τις επιδόσεις G1, G2, G3
inputs = ["G1", "G2", "G3"]
ga = students[inputs]

# Κανονικοποίηση
scaler = StandardScaler()
scaled_ga = scaler.fit_transform(ga)

# Ζήτα από τον χρήστη τον αριθμό ομάδων
k = int(input("\nΔώσε αριθμό ομάδων (clusters): "))
#ομαδοποίηση
kmeans = KMeans(n_clusters=k, random_state=0)
students["Cluster"] = kmeans.fit_predict(scaled_ga)

# Εμφάνιση στατιστικών ανά ομάδα
print("\nΣτατιστικά κάθε ομάδας")
for group in range(k):
    subset = students[students["Cluster"] == group]
    print(f"\nΟμάδα {group}:")
    print(subset[inputs].describe())

# Γραφικό περιβάλλον στατιστικών
plt.scatter(students["G1"], students["G2"], c=students["Cluster"])
plt.xlabel("G1")
plt.ylabel("G2")
plt.title("Ομαδοποίηση Φοιτητών")
plt.show()

# Εισαγωγή νέου φοιτητή
print("\nΕισαγωγή νέου φοιτητή,δώσε βαθμούς")
new_g1 = float(input("G1: "))
new_g2 = float(input("G2: "))
new_g3 = float(input("G3: "))

new_student = pd.DataFrame([[new_g1, new_g2, new_g3]], columns=inputs)

# Κανονικοποίηση
scaled_new = scaler.transform(new_student)

# Πρόβλεψη ομάδας
new_cluster = kmeans.predict(scaled_new)[0]
print(f"\nΟ νέος φοιτητής ανήκει στην ομάδα: {new_cluster}")

# Εύρεση κοντινότερων φοιτητών με KNN
same_cluster = students[students["Cluster"] == new_cluster][inputs]

scaled_same = scaler.transform(same_cluster)

neighbors = NearestNeighbors(n_neighbors=5)
neighbors.fit(scaled_same)

distances, nearest_indices = neighbors.kneighbors(scaled_new)

print("\nΚοντινότεροι φοιτητές στην ίδια ομάδα")
closest_students = same_cluster.iloc[nearest_indices[0]]
print(closest_students)
