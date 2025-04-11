# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import os
import timeit

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
# Algorithme de Dijkstra
def Dijkstra(point_depart, point_arrivee):
    # Vérifie que les sommets existent dans le dictionnaire
    if point_depart not in dicSuccDist or point_arrivee not in dicSuccDist:
        raise ValueError("Point de départ ou d'arrivée non valide.")

    # Initialisation des distances à l'infini pour tous les sommets
    distances = {point: float('inf') for point in dicSuccDist}
    distances[point_depart] = 0  # Distance du point de départ à lui-même = 0

    # Dictionnaire pour mémoriser le prédécesseur de chaque sommet
    precedent = {point: None for point in dicSuccDist}

    # Liste des sommets non encore visités
    non_visites = list(dicSuccDist.keys())

    while non_visites:
        # Sélectionne le sommet non visité avec la plus petite distance connue
        sommet_actuel = min(non_visites, key=lambda p: distances[p])

        # Si la plus petite distance restante est infinie, on ne peut plus avancer
        if distances[sommet_actuel] == float('inf'):
            break

        # Si on atteint le point d’arrivée, on peut sortir de la boucle
        if sommet_actuel == point_arrivee:
            break

        # Marque ce sommet comme visité
        non_visites.remove(sommet_actuel)

        # Explore tous les voisins du sommet actuel
        for voisin, poids in dicSuccDist.get(sommet_actuel, []):
            # Calcule la nouvelle distance via le sommet actuel
            nouvelle_distance = distances[sommet_actuel] + poids

            # Si cette nouvelle distance est plus courte, on la met à jour
            if nouvelle_distance < distances[voisin]:
                distances[voisin] = nouvelle_distance
                precedent[voisin] = sommet_actuel

    # Reconstruction du chemin à partir du dictionnaire des précédents
    chemin = []
    courant = point_arrivee
    while courant is not None:
        chemin.insert(0, courant)  # Insère chaque sommet au début de la liste
        courant = precedent[courant]

    # Si aucune distance n'a été trouvée pour le point d'arrivée
    if distances[point_arrivee] == float('inf'):
        return [], float('inf')

    # Retourne le chemin trouvé et sa distance
    return chemin, distances[point_arrivee]


# Fonction de test pour Dijkstra
def test_dijkstra():
    chemin, distance = Dijkstra(255402679, 2436291971)
    print(f"Chemin trouvé (Dijkstra): {chemin}")
    print(f"Distance totale (Dijkstra): {distance}")

# Mesurer le temps d'exécution de Dijkstra
temps_execution = timeit.timeit('test_dijkstra()', globals=globals(), number=1)
print(f"Temps d'exécution de Dijkstra : {temps_execution:.6f} secondes")

# ------------------------------
# Algorithme de Bellman-Ford-Kalaba (BFK)

# Algorithme de Bellman-Ford-Kalaba (BFK)
def BFK(id_dep, id_arriv):
    # Initialisation des distances à l'infini pour tous les sommets
    distances = {s: float('inf') for s in dicSuccDist}
    distances[id_dep] = 0  # La distance du sommet de départ à lui-même est 0

    # Dictionnaire pour mémoriser le prédécesseur de chaque sommet (pour reconstruire le chemin)
    previous = {s: None for s in dicSuccDist}

    # Relaxation de toutes les arêtes |V| - 1 fois (V = nombre de sommets)
    for _ in range(len(dicSuccDist) - 1):
        # Pour chaque sommet du graphe
        for u in dicSuccDist:
            # Pour chaque voisin de u avec sa distance (pondération)
            for v, weight in dicSuccDist[u]:
                # Si un chemin plus court est trouvé, on le met à jour
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    previous[v] = u

    # Reconstruction du plus court chemin depuis id_arriv en remontant les prédécesseurs
    chemin = []
    current = id_arriv
    while current is not None:
        chemin.insert(0, current)  # On insère les sommets au début du chemin
        current = previous[current]  # On remonte vers le prédécesseur

    # Retourne la liste des sommets constituant le chemin et la distance totale
    return chemin, distances[id_arriv]

# Fonction de test pour BFK
def test_bfk():
    chemin, distance = BFK(388382398, 1888303671)
    print(f"Chemin trouvé (BFK): {chemin}")
    print(f"Distance totale (BFK): {distance}")

# Mesure du temps d'exécution de BFK
temps_bfk = timeit.timeit('test_bfk()', globals=globals(), number=1)
print(f"Temps d'exécution de Bellman-Ford-Kalaba : {temps_bfk:.6f} secondes")

# ------------------------------
# Comparaison des deux algorithmes

print(f"Comparaison entre les 2 temps : {temps_bfk:.6f} secondes pour BFK et {temps_execution:.6f} secondes pour Dijkstra.")

if temps_bfk < temps_execution:
    print("Le meilleur algorithme est Bellman-Ford-Kalaba")
else:
    print("Le meilleur algorithme est Dijkstra")

