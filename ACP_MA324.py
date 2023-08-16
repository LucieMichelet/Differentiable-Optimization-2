# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:15:56 2022

@author: Lucie
"""


import numpy as np
import matplotlib.pyplot as plt
import copy



def esperance(Xi):
    """
    fonction qui calcul un estimateur sans biais de l'esperence d'un vecteur Xi
    :param Xi: un vecteur de taille (m,)
    :return: l'esperence du vecteur Xi
    """

    m = Xi.shape[0]
    return np.sum(Xi) / m


def variance(Xi):
    """
    fonction qui calcul un estimateur avec un biais asymptotique de la variance d'un vecteur Xi
    :param Xi: un vecteur de taille (m,)
    :return: la variance du vecteur Xi
    """

    m = Xi.shape[0]
    Xi_bar = esperance(Xi)
    return np.sum((Xi - Xi_bar)**2) / (m - 1)


def centre_red(R):
    """
    fonction qui a partir une variable aleatoire R de loi differente, centre et reduit les Xi pour qu'ils soient plus
    homogène a etudier
    :param R: un vecteur aléatoire de taille (m, n)
    :return Rcr: le vecteur aléatoire R modifié de facon que l'esperance de Xi soient 0 et leurs variance 1
    """

    # on récupère les dimensions de R pour créer la matrice résultat Rcr de même dimension
    m, n = R.shape
    Rcr = np.zeros((m, n))

    # pour chaque colonne de R on centre et réduit indépendamment les données car les Xi ne suivent pas les mêmes lois
    for i in range(n):
        Xi = R[:, i]

        # on calcule l'espérence et la variance de chaque Xi
        E = esperance(Xi)
        var = variance(Xi)

        # on "décale" les données de chaque colonnes indépendamment
        Rcr[:, i] = (Xi - E) / np.sqrt(var)

    # on retourne la matrice Rcr qui contient les données centrées et réduites
    return Rcr


def approx(R, k):
    """
    le but de la fonction est de decomposé en vecteur propre la matrice R suivant k direction que l'on doit déterminer,
    apres avoir projeté R sur ces vecteurs la matrice résultante sera dans un EV de dimension plus faible donc
    possiblement affichable sur un plan
    :param R: le vecteur aléatoire, matrice de données de taille (m, n)
    :param k: le nombre de dimension dans lequel on souhaite projeter R
    :return proj: un matrice de taille (m, k)
    """

    # on récupère les dimensions de la matrice R et on créer la matrice proj qui contiendra le résultat
    m, n = R.shape
    proj = np.zeros((m, k))

    # on centre et réduit le vecteur aléatoire R pour que les composantes soient toutes homogènes
    Rcr = centre_red(R)

    # on fait la décomposition SVD de la matrice Rcr, le vecteur U de taille (m, n) et s de taille (n, 1) nous interesse
    # U est une base othonormé de Rcr
    # s contient les valeurs singulière / variance de Rcr trier par ordre décroissante d'importance
    U, s, VT = np.linalg.svd(Rcr)
    u = U[:, :k]

    # on concerve dans proj uniquement les k-composantes les plus importante de la nouvelle base U (variances
    # les plus élevés) : sigma**2 * uj
    for j in range(k):
        proj[:, j] = (s[j]**2) * u[:, j]

    return proj




def ACP2D(R, labelsligne, labelscolonne):
    """
    parameter R : le tableau de données numériques
    parameter labelsligne : noms des lignes (s’ils existent, si non on prendra un vecteur d’entier de 1 à m),
    parameter labelscolonne : noms des colonnes (s’ils existent si non on prendra un vecteur d’entier de 1 à n),
    return :  le graphe qui représente les valeurs des variances σk² et le graphe qui représente le pourcentage de l’explication de la variance de chaque k−composante principal

    """
    m, n = R.shape
    Rcr = centre_red(R)
    U, s, VT = np.linalg.svd(Rcr)
    V = VT.T
    v = V[:, :n]
    Y = Rcr@v
    
        

    Yi = []
    for i in range(1, n+1):
        Yi.append('Y{}'.format(i))


    #Affichage graphe contour
    

    
    Y = np.linspace(0,150) 
    X = np.linspace(0, 150) 
  
    #[X, Y] = np.meshgrid(feature_x, feature_y) 
    

    # Affichage des deux seconds graphqiues

    fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 6))

    Proj = approx(R, 2)
    ax3.scatter(Proj[:, 0], Proj[:, 1], c=labelsligne.astype('float'))
    ax3.set_ylabel('Y2')
    ax3.set_xlabel('Y1')
    ax3.set_title('Analyse en composantes principales pour k = 2')
    
    ax4.contour(Proj,colors=labelsligne.astype('float'))

    

    plt.show()
    

def GPC(A,b,x0,epsilon) : 
    x=copy.copy(x0)
    d=b-A@x
    compteur=0
    y=x+epsilon*np.ones(np.shape(x))
    while np.linalg.norm(x-y) >epsilon and compteur <1000 : 
         y=copy.copy(x)
         t=-(d.T @ (A@x-b))/(d.T @A @d)
         x=x+t*d
         beta=(d.T @ A @ (A@x-b)) / (d.T @ A @ d)
         d=-(A@x-b)+beta*d
         compteur+=1
    print("GPC : La convergence à {} près est obtenue pour {} itérations.".format(epsilon,compteur))
    return x  


def f(x,W):
    y = np.vstack((x,1))
    val = y.T@W
    return np.argmax(val)


    
