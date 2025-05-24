import pandas as pd
import graphics as gr
from graphics import *


def dessiner_points(chemin_image="BAYONNE25.png"):
    dfSommets = pd.read_table('donneesS202/sommets.csv', sep=';', index_col=0)
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


def main():
    dessiner_points()

main()