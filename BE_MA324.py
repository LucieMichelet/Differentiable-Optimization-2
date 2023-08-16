# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:13:41 2022

@author: Lucie
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

#%% ---------------------------------------importation des données----------------------------------------

Databrut = np.genfromtxt("iris.csv", dtype = str, delimiter=',')
Datac = Databrut[0,:-1]                                                   #sepal_length,sepal_width,petal_length,petal_width
Datal = Databrut[1:,-1].astype('float')                                                 #espece
Data = Databrut[1:,:-1].astype('float')

#%%

def trace(X,Y):
    return np.trace(X.T@Y)
    

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



def correlationdirprinc(R, k):
    """
    parameter R : tableau de données numérique de taille [m;n]
    parameter k : entier inférieur à n
    return Cor : matrice correlation de taille [k:n]

    """
    m, n = R.shape
    Cor = np.zeros((k, n))

    Rcr = centre_red(R)
    U, s, VT = np.linalg.svd(Rcr)
    V = VT.T
    v = V[:, :n]
    Y = Rcr@v
    

    for j in range(k):
        Yk = Y[:, j]
        for i in range(n):
            Xi = Rcr[:, i]
            Xi = Xi.reshape(m, 1)
            Cor[j, i] = (Yk.T@Xi/(np.sqrt(variance(Yk))))

    return Cor


def ACP(R, labelsligne, labelscolonne, k=0, epsilon=10**-1):

    m, n = np.shape(R)
    Rcr = centre_red(R)
    U, s, VT = np.linalg.svd(Rcr)
    V = VT.T
    v = V[:, :n]
    Y = Rcr@v
    
    for i in range(n):
        if variance(Y[:,i])>=1-epsilon :
            k +=1
        elif variance(Y[:,i+1])<1-epsilon :
            break
    
    return Y,s,k
    
    
    


#%%

D = Data

m,d = np.shape(Data)
x = np.ones((m,d+1))
c = 3
B = np.ones((m,c))

#création de la matrice B
for i in range(m):
    x[i,:-1] = Data[i,:]
    for j in range(c):
        if (Datal[i] == j) :
            B[i,j] = 1
        else :
            B[i,j] = 0 
            
x = x.T
w0 = np.ones((d,c))
rho = 1     #verifier la valeur
A = D.T@D - rho*np.eye(d)
C = D.T@B

#Création de la matrice W
W = np.zeros((d+1,c))
def app_global_MC(train_data,train_labels, nbcat,rho,epsilon):
    for n in range(c) :                                                   #on calcul b
        W2 = GPC(A,C,w0,10**(-10)).reshape(200,1)                              #on calcul un+2
        W[:,n+2]=W2     
    return W
    
Wf = app_global_MC(Data, Datal, c, 1, 10**(-10))

#%% Partie 1

a,b,k = ACP(Data,Datal,Datac)
plt.contour(a,b,k)


#%% Partie 2

