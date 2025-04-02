"""
Crée le 31/03/2025
Auteur : Hoarau Erwan, Lalanne Victor
"""

#importation des modules
import pandas as pd
import numpy as np
import os  

os.chdir("C:\\Users\\Hoarau\\Desktop\\cours\\2eme_semestre\\S2.02\\")

aretes = pd.read_csv("aretes.csv", sep=";", decimal=",", index_col="id_arete", encoding="latin1")
points = pd.read_csv("points.csv", sep=";", decimal=",", index_col="id_points", encoding="latin1")

#transposition et conversion en dictionnaire
dic_aretes = aretes.T.to_dict()
dic_points = points.T.to_dict()

#création de la liste des points dans les aretes
pointsInAretes = []

# Ajoute uniquement le premier et le dernier sommet de chaque arrete
for way in dic_aretes.values():
    temp = [i.strip() for i in way['lstpoints'][1:-1].split(',')]
    if len(temp) >= 2:  # Vérifie qu'il y a au moins deux sommets
        if temp[0] not in pointsInAretes:
            pointsInAretes.append(temp[0])
        if temp[-1] not in pointsInAretes:
            pointsInAretes.append(temp[-1])

# Supprime les sommets de dic_points qui ne font pas partie d'un arc
for sommet in list(dic_points.keys()):
    if str(sommet) not in pointsInAretes:
        del dic_points[sommet]
        
del sommet, temp, way #suppression des variables innutiles


matriceAdjacence = pd.DataFrame(index=dic_points.keys(), columns=dic_points.keys())