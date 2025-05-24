# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json
import os
import timeit
import heapq
import math
from math import pi, acos, sin, cos, radians

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
# Heuristique 
def heuristique(a, b): # fonction distanceGPS modifiée 
    

    # Récupération des coordonnées depuis le DataFrame
    latA = dfSommets.loc[a, 'lat']
    lonA = dfSommets.loc[a, 'lon']
    latB = dfSommets.loc[b, 'lat']
    lonB = dfSommets.loc[b, 'lon']

    # Conversion en radians
    ltA = radians(latA)
    ltB = radians(latB)
    loA = radians(lonA)
    loB = radians(lonB)

    # Rayon de la Terre en mètres
    RT = 6378137
    # Formule du grand cercle
    S = acos(round(sin(ltA) * sin(ltB) + cos(ltA) * cos(ltB) * cos(abs(loB - loA)), 14))

    return S * RT  # distance en mètres



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
    chemin, distance = Dijkstra(388382398, 1888303671)
    print(f"Chemin trouvé (Dijkstra): {chemin}")
    print(f"Distance totale (Dijkstra): {distance}")

# Mesurer le temps d'exécution de Dijkstra
temps_execution = timeit.timeit('test_dijkstra()', globals=globals(), number=1)
print(f"Temps d'exécution de Dijkstra : {temps_execution:.6f} secondes")

# ------------------------------
# Algorithme de Bellman
def Bellman(id_dep, id_arriv):
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

# Fonction de test pour Bellman
def test_bellman():
    chemin, distance = Bellman(388382398, 1888303671)
    print(f"Chemin trouvé (Bellman): {chemin}")
    print(f"Distance totale (Bellman): {distance}")

# Mesure du temps d'exécution de Bellman
temps_bellman = timeit.timeit('test_bellman()', globals=globals(), number=1)
print(f"Temps d'exécution de Bellman : {temps_bellman:.6f} secondes")



# ------------------------------
# Algorithme de Bellman-Ford-Kalaba
def BFK(depart, arrive):
    n = len(dicSucc)
    listeDist = [float("inf")] * n
    listeDist[correspSomInd[depart]] = 0
    a_traiter = [depart]
    pred = [None] * n
    for _ in range(len(dicSucc)):
        a_traiterFutur = []
        for som in a_traiter:
            somInd = correspSomInd[som]
            for succ, poids in dicSuccDist[som]:
                succInd = correspSomInd[succ]
                if listeDist[succInd] > listeDist[somInd] + poids:
                    listeDist[succInd] = listeDist[somInd] + poids
                    pred[succInd] = som
                    a_traiterFutur.append(succ)
        a_traiter = a_traiterFutur

    chemin = []
    courant = arrive
    while courant is not None:
        chemin.insert(0, courant)
        courant = pred[correspSomInd[courant]]

    if listeDist[correspSomInd[arrive]] == float('inf'):
        return [depart], float('inf')
    return chemin,listeDist[correspSomInd[arrive]]


# Fonction de test pour BFK
def test_bfk():
    chemin, distance = BFK(388382398, 1888303671)
    print(f"Chemin trouvé (BFK): {chemin}")
    print(f"Distance totale (BFK): {distance}")

# Mesurer le temps d'exécution de BFK
temps_bfk = timeit.timeit('test_bfk()', globals=globals(), number=1)
print(f"Temps d'exécution de BFK : {temps_bfk:.6f} secondes")


# ------------------------------
# Algorithme A*
def A_Star(point_depart, point_arrivee):
    if point_depart not in dicSuccDist or point_arrivee not in dicSuccDist:
        raise ValueError("Point de départ ou d'arrivée non valide.")

    # Initialisation
    distances = {point: float('inf') for point in dicSuccDist}
    distances[point_depart] = 0
    precedent = {point: None for point in dicSuccDist}
    f_scores = {point: float('inf') for point in dicSuccDist}
    f_scores[point_depart] = heuristique(point_depart, point_arrivee)

    non_visites = list(dicSuccDist.keys())

    while non_visites:
        # Chercher le sommet non visité avec la plus petite f(n)
        sommet_actuel = min(non_visites, key=lambda p: f_scores[p])

        if f_scores[sommet_actuel] == float('inf'):
            break  # Aucun chemin disponible

        if sommet_actuel == point_arrivee:
            break  # On a trouvé le chemin

        non_visites.remove(sommet_actuel)

        # Pour chaque voisin du sommet actuel
        for voisin, poids in dicSuccDist.get(sommet_actuel, []):
            tentative_g = distances[sommet_actuel] + poids
            f_voisin = tentative_g + heuristique(voisin, point_arrivee)

            if tentative_g < distances[voisin]:
                distances[voisin] = tentative_g
                precedent[voisin] = sommet_actuel
                f_scores[voisin] = f_voisin

    # Reconstruction du chemin
    
    chemin = []
    courant = point_arrivee
    while courant is not None:
        chemin.insert(0, courant)
        courant = precedent[courant]

    if distances[point_arrivee] == float('inf'):
        return [], float('inf')

    return chemin, distances[point_arrivee]


# Fonction de test pour A*
def test_a_star():
    chemin, distance = A_Star(255402679, 2436291971)
    print(f"Chemin trouvé (A*) : {chemin}")
    print(f"Distance totale (A*) : {distance:.2f} mètres")

# Mesurer le temps d'exécution de A*
temps_a_star = timeit.timeit('test_a_star()', globals=globals(), number=1)  
print(f"Temps d'exécution de A* : {temps_a_star:.6f} secondes")

# --------------------------------
# Algorithme de dijkstra mon ami (a garder poyur plus tard)
def dijkstra_mon_ami(depart, arrive):
    n = len(dicSucc)
    listeDist = [float("inf")] * n
    pred = [None] * n

    depInd = correspSomInd[depart]
    arrInd = correspSomInd[arrive]
    listeDist[depInd] = 0

    ouvert = []  # File de priorité pour gérer les sommets à explorer
    heapq.heappush(ouvert, (0, depart))  # On commence par le sommet de départ avec une priorité de 0


    while ouvert:
        _, courant = heapq.heappop(ouvert)  # Extraire le sommet avec la plus petite distance estimée
        courantInd = correspSomInd[courant]

        if courant == arrive:
            break

        for succ, poids in dicSuccDist[courant]:
            succInd = correspSomInd[succ]
            tentative = listeDist[courantInd] + poids  # Calcul de la distance temporaire par ce chemin
            if tentative < listeDist[succInd]:  # Si c’est un chemin plus court que celui déjà enregistré alors :
                listeDist[succInd] = tentative  # Mise à jour de la distance
                pred[succInd] = courant  # Mise à jour du prédécesseur pour reconstruire le chemin plus tard
                heapq.heappush(ouvert, (tentative, succ))  # On ajoute ce sommet à la file de priorité

    # Reconstruction du chemin
    chemin = []
    courant = arrive
    while courant is not None:
        chemin.insert(0, courant) #on remonte le chemin des prédécesseurs au départ
        courant = pred[correspSomInd[courant]]

    if listeDist[arrInd] == float('inf'):
        return [depart], float('inf')
    return chemin, listeDist[arrInd]

#Fonction de test pour dijkstra_mon_ami
def test_dijkstra_mon_ami():
    chemin, distance = dijkstra_mon_ami(388382398, 1888303671)
    print(f"Chemin trouvé (dijkstra_mon_ami): {chemin}")
    print(f"Distance totale (dijkstra_mon_ami): {distance}")

# Mesurer le temps d'exécution de dijkstra_mon_ami
temps_dijkstra_mon_ami = timeit.timeit('test_dijkstra_mon_ami()', globals=globals(), number=1)
print(f"Temps d'exécution de dijkstra_mon_ami : {temps_dijkstra_mon_ami:.6f} secondes")



# ------------------------------
# Comparaison des 4 algorithmes

print(f"Comparaison entre les 4 temps : {temps_bellman:.6f} secondes pour Bellman, {temps_bfk:.6f} pour BFK, {temps_execution:.6f} secondes pour Dijkstra et {temps_a_star:.6f} pour A*.")

if temps_bellman <= temps_bfk and temps_bellman <= temps_execution and temps_bellman <= temps_a_star:
    print("Le meilleur algorithme est Bellman")
elif temps_bfk <= temps_bellman and temps_bfk <= temps_execution and temps_bfk <= temps_a_star:
    print("Le meilleur algorithme est BFK")
elif temps_execution <= temps_bellman and temps_execution <= temps_bfk and temps_execution <= temps_a_star:
    print("Le meilleur algorithme est Dijkstra")
else:
    print("Le meilleur algorithme est A*")

