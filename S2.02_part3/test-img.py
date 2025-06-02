import pandas as pd
import numpy as np
import json
import os
import timeit
import heapq
import math
from math import pi, acos, sin, cos, radians
import graphics as gr
from graphics import *

# Répertoire par défaut à modifier :
os.chdir(r'C:\Users\Hoarau\Desktop\cours\2eme_semestre\S2.02\S2.02_part3\donneesS202')  

# Dictionnaire des successeurs
with open("dicsucc.json", "r") as fichier: # Chargement du fichier JSON
    dicSuccCleStr = json.load(fichier) # Conversion des clés de chaîne en entiers
dicSucc = {int(k): v for k, v in dicSuccCleStr.items()} # Conversion des clés de chaîne en entiers
del dicSuccCleStr # Libération de la mémoire

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



# Créer au départ pour avoir une idée de comment dessiner les points sur une image
def dessiner_points(chemin_image="C:/Users/Hoarau/Desktop/cours/2eme_semestre/S2.02/S2.02_part3/BAYONNE25.png"):
    dfSommets = pd.read_table('C:/Users/Hoarau/Desktop/cours/2eme_semestre/S2.02/S2.02_part3/donneesS202/sommets.csv', sep=';', index_col=0)
    min_lat, max_lat = 43.482630, 43.506698
    min_lon, max_lon = -1.493282, -1.454422
    largeur_image, hauteur_image = 1130, 969

    fenetre = gr.GraphWin("Carte avec points", largeur_image, hauteur_image)
    image = gr.Image(gr.Point(largeur_image / 2, hauteur_image / 2), chemin_image)
    image.draw(fenetre)

    # Pour chaque sommet, on trace un cercle rouge
    for _, row in dfSommets.iterrows():
        lat = row['lat']
        lon = row['lon']
        x, y = pos_to_pix(lat, lon, min_lat, max_lat, min_lon, max_lon, largeur_image, hauteur_image)
        point = gr.Circle(gr.Point(x, y), 2)
        point.setFill('red')
        point.draw(fenetre)

    fenetre.getMouse()
    fenetre.close()

    
def pos_to_pix(lat, lon, min_lat=43.482630, max_lat=43.506698,
               min_lon=-1.493282, max_lon=-1.454422,
               largeur_image=1130, hauteur_image=969):
    # Produit en croix pour la longitude (x)
    x = (lon - min_lon) / (max_lon - min_lon) * largeur_image
    
    # Produit en croix pour la latitude (y), inversé car y=0 est en haut sur une image
    y = (max_lat - lat) / (max_lat - min_lat) * hauteur_image
    
    return x, y

# ------------------------------
# Fonction dessiner graphe qui reprend la logique de dessiner_points
def dessiner_graphe(chemin_image="C:/Users/Hoarau/Desktop/cours/2eme_semestre/S2.02/S2.02_part3/BAYONNE25.png"):
    dfSommets = pd.read_table('C:/Users/Hoarau/Desktop/cours/2eme_semestre/S2.02/S2.02_part3/donneesS202/sommets.csv', sep=';', index_col=0)
    min_lat, max_lat = 43.482630, 43.506698
    min_lon, max_lon = -1.493282, -1.454422
    largeur_image, hauteur_image = 1130, 969

    fenetre = gr.GraphWin("Graphe", largeur_image, hauteur_image)
    image = gr.Image(gr.Point(largeur_image / 2, hauteur_image / 2), chemin_image)
    image.draw(fenetre)

    # Calcul des positions
    positions_pix = {}
    for sommet, row in dfSommets.iterrows():
        x, y = pos_to_pix(row['lat'], row['lon'], min_lat, max_lat, min_lon, max_lon, largeur_image, hauteur_image)
        positions_pix[sommet] = (x, y)
        cercle = gr.Circle(gr.Point(x, y), 2)
        cercle.setFill("red")
        cercle.draw(fenetre)

    # Dessin des arcs
    lignes = {}
    for u in dicSucc:
        for v, _ in dicSuccDist[u]:
            if (u, v) not in lignes:  # éviter les doublons
                x1, y1 = positions_pix[u]
                x2, y2 = positions_pix[v]
                ligne = gr.Line(gr.Point(x1, y1), gr.Point(x2, y2))
                ligne.setFill("gray")
                ligne.draw(fenetre)
                lignes[(u, v)] = ligne

    return fenetre, positions_pix, lignes


# -------------------------------
# Dijkstra visuel
import time

def Dijkstra_visuel(depart, arrivee, fenetre, positions_pix, lignes):
    distances = {point: float('inf') for point in dicSuccDist}
    distances[depart] = 0
    precedent = {point: None for point in dicSuccDist}
    non_visites = list(dicSuccDist.keys())

    while non_visites:
        u = min(non_visites, key=lambda x: distances[x])
        if distances[u] == float('inf') or u == arrivee:
            break
        non_visites.remove(u)
        for v, poids in dicSuccDist[u]:
            d = distances[u] + poids
            if d < distances[v]:
                distances[v] = d
                precedent[v] = u

                # Colorier l’arc visité
                if (u, v) in lignes:
                    lignes[(u, v)].setFill("yellow")
                    lignes[(u, v)].setWidth(3)
                elif (v, u) in lignes:  # graphe non orienté
                    lignes[(v, u)].setFill("yellow")
                    lignes[(v, u)].setWidth(3)

                # Pause pour visualisation
                time.sleep(0.02)

    # Reconstruction du chemin final
    chemin = []
    courant = arrivee
    while courant is not None:
        chemin.insert(0, courant)
        courant = precedent[courant]

    # Colorer le chemin final en bleu
    for i in range(len(chemin) - 1):
        u, v = chemin[i], chemin[i+1]
        if (u, v) in lignes:
            lignes[(u, v)].setFill("blue")
            lignes[(u, v)].setWidth(5)
        elif (v, u) in lignes:
            lignes[(v, u)].setFill("blue")
            lignes[(v, u)].setWidth(5)
        time.sleep(0.07)

    return chemin, distances[arrivee]



# -------------------------------
# Bellman visuel

def Bellman_visuel(depart, arrivee, fenetre, positions_pix, lignes):
    distances = {s: float('inf') for s in dicSuccDist}
    distances[depart] = 0
    previous = {s: None for s in dicSuccDist}

    for _ in range(len(dicSuccDist) - 1):
        for u in dicSuccDist:
            for v, weight in dicSuccDist[u]:
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    previous[v] = u

                    if (u, v) in lignes:
                        lignes[(u, v)].setFill("purple")
                        lignes[(u, v)].setWidth(3)
                    elif (v, u) in lignes:
                        lignes[(v, u)].setFill("purple")
                        lignes[(v, u)].setWidth(3)



    chemin = []
    current = arrivee
    while current is not None:
        chemin.insert(0, current)
        current = previous[current]

    for i in range(len(chemin) - 1):
        u, v = chemin[i], chemin[i+1]
        if (u, v) in lignes:
            lignes[(u, v)].setFill("pink")
            lignes[(u, v)].setWidth(5)
        elif (v, u) in lignes:
            lignes[(v, u)].setFill("pink")
            lignes[(v, u)].setWidth(5)
        time.sleep(0.07)

    return chemin, distances[arrivee]



# -------------------------------
# Algorithme de Bellman-Ford-Kalaba visuel

def BFK_visuel(depart, arrivee, fenetre, positions_pix, lignes):
    n = len(dicSucc)
    listeDist = [float("inf")] * n
    listeDist[correspSomInd[depart]] = 0
    a_traiter = [depart]
    pred = [None] * n

    for _ in range(n):
        a_traiter_futur = []
        for u in a_traiter:
            u_ind = correspSomInd[u]
            for v, poids in dicSuccDist[u]:
                v_ind = correspSomInd[v]
                if listeDist[v_ind] > listeDist[u_ind] + poids:
                    listeDist[v_ind] = listeDist[u_ind] + poids
                    pred[v_ind] = u
                    a_traiter_futur.append(v)

                    if (u, v) in lignes:
                        lignes[(u, v)].setFill("red")
                        lignes[(u, v)].setWidth(3)
                    elif (v, u) in lignes:
                        lignes[(v, u)].setFill("red")
                        lignes[(v, u)].setWidth(3)

        a_traiter = a_traiter_futur

    # Reconstruction du chemin
    chemin = []
    courant = arrivee
    while courant is not None:
        chemin.insert(0, courant)
        courant = pred[correspSomInd[courant]]

    # Colorer le chemin final
    for i in range(len(chemin) - 1):
        u, v = chemin[i], chemin[i + 1]
        if (u, v) in lignes:
            lignes[(u, v)].setFill("orange")
            lignes[(u, v)].setWidth(5)
        elif (v, u) in lignes:
            lignes[(v, u)].setFill("orange")
            lignes[(v, u)].setWidth(5)
        time.sleep(0.07)

    return chemin, listeDist[correspSomInd[arrivee]]


# -------------------------------
# A* visuel

def A_Star_visuel(depart, arrivee, fenetre, positions_pix, lignes):
    if depart not in dicSuccDist or arrivee not in dicSuccDist:
        raise ValueError("Départ ou arrivée invalide")

    distances = {p: float('inf') for p in dicSuccDist}
    distances[depart] = 0
    precedent = {p: None for p in dicSuccDist}
    f_scores = {p: float('inf') for p in dicSuccDist}
    f_scores[depart] = heuristique(depart, arrivee)

    non_visites = list(dicSuccDist.keys())

    while non_visites:
        u = min(non_visites, key=lambda x: f_scores[x])
        if f_scores[u] == float('inf') or u == arrivee:
            break
        non_visites.remove(u)

        for v, poids in dicSuccDist[u]:
            tentative_g = distances[u] + poids
            f_v = tentative_g + heuristique(v, arrivee)

            if tentative_g < distances[v]:
                distances[v] = tentative_g
                precedent[v] = u
                f_scores[v] = f_v

                if (u, v) in lignes:
                    lignes[(u, v)].setFill("green")
                    lignes[(u, v)].setWidth(3)
                elif (v, u) in lignes:
                    lignes[(v, u)].setFill("green")
                    lignes[(v, u)].setWidth(3)

                time.sleep(0.02)

    chemin = []
    courant = arrivee
    while courant is not None:
        chemin.insert(0, courant)
        courant = precedent[courant]

    for i in range(len(chemin) - 1):
        u, v = chemin[i], chemin[i+1]
        if (u, v) in lignes:
            lignes[(u, v)].setFill("lightgreen")
            lignes[(u, v)].setWidth(5)
        elif (v, u) in lignes:
            lignes[(v, u)].setFill("lightgreen")
            lignes[(v, u)].setWidth(5)
        time.sleep(0.07)

    return chemin, distances[arrivee]


def main():
    fenetre, positions_pix, lignes = dessiner_graphe()
    debut = time.perf_counter()
    chemin, dist = Dijkstra_visuel(388382398, 1888303671, fenetre, positions_pix, lignes)
    fin = time.perf_counter()
    print("Chemin Dijkstra :", chemin)
    print("Distance Dijkstra :", dist)
    print(f"Temps d'exécution de Dijkstra visuel : {fin - debut:.6f} secondes")
    
    debut = time.perf_counter()
    chemin, dist = Bellman_visuel(388382398, 1888303671, fenetre, positions_pix, lignes)
    fin = time.perf_counter()
    print("Chemin Bellman :", chemin)
    print("Distance Bellman :", dist)
    print(f"Temps d'exécution de Bellman visuel : {fin - debut:.6f} secondes")

    debut = time.perf_counter()
    chemin, dist = BFK_visuel(388382398, 1888303671, fenetre, positions_pix, lignes)
    fin = time.perf_counter()
    print("Chemin BFK :", chemin)
    print("Distance BFK :", dist)
    print(f"Temps d'exécution de BFK visuel : {fin - debut:.6f} secondes")

    debut = time.perf_counter()
    chemin, dist = A_Star_visuel(388382398, 1888303671, fenetre, positions_pix, lignes)
    fin = time.perf_counter()
    print("Chemin A* :", chemin)
    print("Distance A* :", dist)
    print(f"Temps d'exécution de A* visuel : {fin - debut:.6f} secondes")
    fenetre.getMouse()
    fenetre.close()

main()