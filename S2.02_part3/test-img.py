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



def main():
    fenetre, positions_pix, lignes = dessiner_graphe()
    fenetre.getMouse()
    fenetre.close()

main()