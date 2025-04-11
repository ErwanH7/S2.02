# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import os
import timeit
import heapq

# Répertoire par défaut à modifier :
os.chdir(r'C:\Users\Hoarau\Desktop\cours\2eme_semestre\S2.02\S2.02_part2\donneesS202')

# Dictionnaire des successeurs
with open("dicsucc.json", "r") as fichier:
    dicSuccCleStr = json.load(fichier)
dicSucc = {int(k): v for k, v in dicSuccCleStr.items()}
del dicSuccCleStr

# Dictionnaire des successeurs avec distance
with open("dicsuccdist.json", "r") as fichier:
    dicSuccDistCleStr = json.load(fichier)
dicSuccDist = {int(k): v for k, v in dicSuccDistCleStr.items()}
del dicSuccDistCleStr

# Chargement des fichiers CSV
dfAretes = pd.read_table('aretes.csv', sep=';', index_col=0)
dfSommets = pd.read_table('sommets.csv', sep=';', index_col=0)
dfMatricePoids = pd.read_csv('matrice_poids.csv', sep=';', index_col=0)
dfMatriceAdj = pd.read_csv('matrice_adjacence.csv', sep=';', index_col=0)

# Correspondances index <-> sommet
correspIndSom = {i: dfMatriceAdj.index[i] for i in range(len(dfMatriceAdj))}
correspSomInd = {ind: dfMatriceAdj.index.get_loc(ind) for ind in dfMatriceAdj.index}

# Conversion des matrices
tabMatAdj = np.array(dfMatriceAdj)
n = len(dfMatriceAdj)
lstMatAdj = [[tabMatAdj[i, j] for j in range(n)] for i in range(n)]

tabMatPoids = np.array(dfMatricePoids)
n = len(tabMatPoids)
lstMatPoids = [[tabMatPoids[i, j] for j in range(n)] for i in range(n)]

# ------------------------------
# Implémentation de Dijkstra 

def Dijkstra(id_dep, id_arriv):
    # Vérification si les sommets sont valides
    if id_dep not in dicSuccDist or id_arriv not in dicSuccDist:
        raise ValueError("Sommet de départ ou d'arrivée invalide.")
    
    # Initialisation des structures de données
    distances = {s: float('inf') for s in dicSuccDist}
    previous = {s: None for s in dicSuccDist}
    distances[id_dep] = 0

    # Liste de priorité pour l'algorithme (utilisation d'un tas binaire)
    heap = [(0, id_dep)]

    while heap:
        # Extraire le sommet avec la plus petite distance
        dist_u, u = heapq.heappop(heap)

        # Si le sommet courant est celui d'arrivée, on peut arrêter l'algorithme
        if u == id_arriv:
            break

        # Mise à jour des voisins du sommet courant
        for v, weight in dicSuccDist.get(u, []):
            alt = dist_u + weight  # Calcul de la distance alternative
            if alt < distances.get(v, float('inf')):  # Si une meilleure distance est trouvée
                distances[v] = alt
                previous[v] = u
                heapq.heappush(heap, (alt, v))  # Ajouter le sommet voisin à la pile

    # Si la distance de l'arrivée est toujours infinie, il n'y a pas de chemin
    if distances[id_arriv] == float('inf'):
        return [], float('inf')

    # Reconstruction du chemin à partir du dictionnaire 'previous'
    chemin = []
    current = id_arriv
    while current is not None:
        chemin.insert(0, current)
        current = previous.get(current)

    return chemin, distances[id_arriv]

# Fonction de test pour l'exécution de Dijkstra
def test_dijkstra():
    chemin, distance = Dijkstra(388382398, 1888303671)
    print(f"Chemin trouvé (Dijkstra): {chemin}")
    print(f"Distance totale (Dijkstra): {distance}")

# Mesurer le temps d'exécution de Dijkstra avec timeit
temps_execution = timeit.timeit('test_dijkstra()', globals=globals(), number=1)

# Affichage du temps d'exécution
print(f"Temps d'exécution de Dijkstra : {temps_execution:.6f} secondes")

# ------------------------------
# Implémentation de Bellman Ford Kalaba (BFK)

# Algorithme de Bellman-Ford-Kalaba
def BFK(id_dep, id_arriv):
    # Initialisation des sommets de départ et d'arrivée
    dep = id_dep
    arr = id_arriv

    # Initialisation des distances à l'infini et des prédécesseurs à None
    distances = {s: float('inf') for s in dicSuccDist}
    previous = {s: None for s in dicSuccDist}
    distances[dep] = 0  # La distance du sommet de départ est 0

    # Relaxation des arêtes |V| - 1 fois (nombre de sommets - 1)
    for _ in range(len(dicSuccDist) - 1):
        for u in dicSuccDist:
            for v, weight in dicSuccDist[u]:
                # Si une distance plus courte est trouvée, on la met à jour
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    previous[v] = u

    # Reconstruction du plus court chemin en remontant les prédécesseurs
    chemin = []
    current = arr
    while current is not None:
        chemin.insert(0, current)  # On insère chaque sommet au début pour obtenir l'ordre correct
        current = previous[current]

    # On retourne le chemin trouvé et la distance minimale
    return chemin, distances[arr]

# Fonction de test pour Bellman-Ford-Kalaba
def test_bfk():
    chemin, distance = BFK(388382398, 1888303671)
    print(f"Chemin trouvé (BFK): {chemin}")
    print(f"Distance totale (BFK): {distance}")

# Mesure du temps d'exécution de BFK avec timeit (une seule exécution)
temps_bfk = timeit.timeit('test_bfk()', globals=globals(), number=1)

# Affichage du temps d'exécution
print(f"Temps d'exécution de Bellman-Ford-Kalaba : {temps_bfk:.6f} secondes")

print(f"Comparaison entre les 2 temps : {temps_bfk:.6f} secondes pour BFK et {temps_execution:.6f} secondes pour Dijkstra.")

if temps_bfk < temps_execution : 
    print ("Le meilleur algorithme est Bellman Ford Kalaba")
else : 
    print ("Le meilleur algorithme est Dijkstra")