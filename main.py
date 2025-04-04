# -*- coding: utf-8 -*-
"""
Crée le 31/03/2025
Auteur : Hoarau Erwan, Lalanne Victor
"""

#importation des modules
import pandas as pd
import numpy as np
from math import acos, pi, sin, cos
import os  

os.chdir("C:\\Users\\Hoarau\\Desktop\\cours\\2eme_semestre\\S2.02\\")

# Chargement des fichiers CSV
aretes = pd.read_csv("aretes.csv", sep=";", decimal=".", encoding="latin1")
points = pd.read_csv("points.csv", sep=";", decimal=".", encoding="latin1")

# Conversion des données en dictionnaires
aretes_dict = aretes.set_index("id_arete").T.to_dict()
points_dict = points.set_index("id_points")[["lat", "lon"]].T.to_dict()

# Liste des ID valides
id_valides = set(points["id_points"].tolist())

# Nettoyage des sommets pour ne garder que les premiers et derniers de chaque arc
pointInArc = []
for way in aretes_dict.values():
    temp = [int(i.strip()) for i in way['lstpoints'][1:-1].split(',') if int(i.strip()) in id_valides]
    if temp:
        if temp[0] not in pointInArc:
            pointInArc.append(temp[0])
        if temp[-1] not in pointInArc:
            pointInArc.append(temp[-1])

# Fonction pour calculer la distance GPS
def distanceGPS(latA, latB, lonA, lonB):
    ltA, ltB = latA * pi / 180, latB * pi / 180
    loA, loB = lonA * pi / 180, lonB * pi / 180
    RT = 6378137  # Rayon de la Terre en mètres
    S = acos(round(sin(ltA) * sin(ltB) + cos(ltA) * cos(ltB) * cos(abs(loB - loA)), 14))
    return S * RT

# Fonction pour calculer la distance entre deux sommets
def distanceSommet(pred, succ):
    for arete in aretes_dict.values():
        points_liste = [int(p.strip()) for p in arete['lstpoints'][1:-1].split(',') if int(p.strip()) in id_valides]
        if points_liste and points_liste[0] == pred and points_liste[-1] == succ:
            return sum(
                distanceGPS(
                    points_dict[points_liste[i]]['lat'],
                    points_dict[points_liste[i+1]]['lat'],
                    points_dict[points_liste[i]]['lon'],
                    points_dict[points_liste[i+1]]['lon']
                ) for i in range(len(points_liste) - 1)
            )
    return float('inf')

# Construction du dictionnaire des successeurs
successeurs = {point: [] for point in pointInArc}
for arete in aretes_dict.values():
    points_liste = [int(p.strip()) for p in arete['lstpoints'][1:-1].split(',') if int(p.strip()) in pointInArc]
    if len(points_liste) >= 2:
        if points_liste[-1] not in successeurs[points_liste[0]]:
            successeurs[points_liste[0]].append(points_liste[-1])
        if points_liste[0] not in successeurs[points_liste[-1]]:
            successeurs[points_liste[-1]].append(points_liste[0])

# Construction de la matrice d'adjacence et de poids
taille = len(pointInArc)
mat_adj = np.zeros((taille, taille), dtype=int)
mat_poids = np.full((taille, taille), float('inf'))

id_list = list(pointInArc)
for pred, succs in successeurs.items():
    for succ in succs:
        pred_idx, succ_idx = id_list.index(pred), id_list.index(succ)
        mat_adj[pred_idx, succ_idx] = 1
        mat_adj[succ_idx, pred_idx] = 1
        dist = distanceSommet(pred, succ)
        mat_poids[pred_idx, succ_idx] = dist
        mat_poids[succ_idx, pred_idx] = dist

# Conversion en DataFrame
mat_adj_df = pd.DataFrame(mat_adj, index=id_list, columns=id_list)
mat_poids_df = pd.DataFrame(mat_poids, index=id_list, columns=id_list)

# Nettoyage des données inutiles
del id_valides, aretes, points, id_list, succ, pred, way, pred_idx, succ_idx, taille, temp, succs, points_liste, dist