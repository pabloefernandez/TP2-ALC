#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

entrenamiento = pd.read_csv("mnist_train.csv",header=None)
test = pd.read_csv("mnist_test.csv",header=None)

#%%
def graficar_im(im_0,n):
    im = im_0.iloc[n,1:]
    im = im.values
    im = im.reshape((28,28))
    plt.imshow(im)
    plt.title(entrenamiento.iloc[n,0])
graficar_im(entrenamiento,1)

def graficar_im_list(imagenes,n):
    imagen = imagenes[n]
    imagen_sin_id = imagen[1:]
    imagen_sin_id = imagen_sin_id.reshape((28,28))
    plt.imshow(imagen_sin_id)
    plt.title(imagen[0])

#%%
#IDea es ver de que numero(id) sos y te sumo a la lista que correspondas

def cantidad_de_digitos(data):
    lista_apariciones = [0]*10
    for i in range(len(data)):
        id = data.iloc[i,0]
        lista_apariciones[id] = lista_apariciones[id] + 1
    return lista_apariciones
a = (cantidad_de_digitos())#Cant de veces que aparece cada digito

def promedio_im():
    lista_con_promedios=[]
    for i in range(10):
        lista_con_promedios.append([np.zeros(785)])
    for i, imagen in enumerate(entrenamiento.iterrows()):
        id = entrenamiento.iloc[i,0]
        lista_con_promedios[id] =np.add(lista_con_promedios[id],imagen[1].values)
    return lista_con_promedios
h = (promedio_im()) # h tiene la suma de todos los vectores(sin dividir)
    

#Sin dividir por el promedio se siguen viendo los numeros de igual manera
def calc_prom(i):
    vector_promedio_con_id = np.divide(h[i], a[i])
    id = vector_promedio_con_id[0][0]
    vector_promedio_sin_id = vector_promedio_con_id[0][1:] #El [0] es solamente pq como se guarda el valor,no deberia estar
    return id, vector_promedio_sin_id

vector_promedio_0 = calc_prom(0)[1]
vector_promedio_1 = calc_prom(1)[1]
vector_promedio_2 = calc_prom(2)[1]
vector_promedio_3 = calc_prom(3)[1]
vector_promedio_4 = calc_prom(4)[1]
vector_promedio_5 = calc_prom(5)[1]
vector_promedio_6 = calc_prom(6)[1]
vector_promedio_7= calc_prom(7)[1]
vector_promedio_8 = calc_prom(8)[1]
vector_promedio_9 = calc_prom(9)[1]

id =  calc_prom(6)[0]

def graficar_prom():
    imagen = vector_promedio_0.reshape((28,28))
    plt.imshow(imagen)
    plt.title(id)

graficar_prom()
    



 
    
 #%%
#%%
"""
ej2
"""
#%%
df = test.iloc[0:200,1:]

def predicciones():
    lista_predicciones = []
    for i, vector in enumerate(df.iterrows()):
        vector = vector[1]
        dist_min = np.linalg.norm(calc_prom(0)[1] - vector)
        for j in range(9): 
             dist = np.linalg.norm(calc_prom(j)[1] - vector)
             if dist < dist_min:
                 dist_min = dist
                 prediccion = calc_prom(j)[0]
        lista_predicciones.append(prediccion) #Voy prediciendo el vector que representa el entrenamiento.
    return lista_predicciones
d = predicciones()
#Me falta hacer una funcion que calcule vectores promedios al darle una i, osea basicamente sacar delgrafico_prom el calculo a una funcion.



def precision():
    contador = 0
    for i, id in enumerate(test.iloc[0:200,0]):
        if id == d[i]:
            contador += 1
    return contador,contador/200
c = precision()



#%%








#%%
def potencia_matriz(B,x,k):#Aca se realiza el metodo de la potencia
  e = pow(10,-5)
  fila = x.shape[0]
  x = x.reshape((1,fila))
  x_anterior = x
  x = x.reshape((fila,1))
  x = B @ x/np.linalg.norm(B @ x)
  x = x.reshape((fila,1))
  while(x_anterior @ x >= (1-e)):
      x_anterior = x.reshape((1,fila))
      x = B @ x/np.linalg.norm(B @ x)
      x = x.reshape((fila,1))
  return x


def cociente(B, x): #Implementacion de Cociente de Rayleigh para aproximar autovalor para cada aproximacion de autovector proviente del metodo de la potencia
    x_transpuesta = np.transpose(x)
    Bx = B @ x
    resultado = (x_transpuesta @ Bx)/(x_transpuesta @ x)
    return resultado

x = np.random.rand(2)
x_norm  = x / np.linalg.norm(x)
A_primera = np.random.rand(3,2)
A = A_primera
At = np.transpose(A)
[U_1,S,Vh] = np.linalg.svd(A)

i,j = (A.shape)
U = np.zeros((i,i))
E = np.zeros((i,j))
V_adjunta = np.zeros((j,j))
for k in range(j): #solo me imteresa el caso m>n pq A es de 10000*784, entonces m > n
    At = np.transpose(A)
    B = At @ A
    autovector = potencia_matriz(B,x,k)
    sigma = np.sqrt(cociente(B,autovector))
    u1 = (A@autovector)/sigma #Calulado todo, vamos a ir formando U, E, V
    U[k] = u1.reshape((u1.shape[0],))
    E[k][k] = sigma
    V_adjunta[k] = autovector.reshape((autovector.shape[0],)) #Ya esta transpuesta
    autovector_traspuesto = autovector.reshape((1,j))
    
    A = A - sigma*(u1.reshape((i,1)) @ autovector_traspuesto)

U = np.transpose(U)

F = U@E@V_adjunta
print(F)
print(A_primera)

#%% 4a
test_2000 = test.iloc[:2000][:].copy()

lista_matrices_M = []
lista_cantidad = cantidad_de_digitos(test_2000)
for i in range(10):
    lista_matrices_M.append(np.zeros((lista_cantidad[i],784)))
lista_contadores = [0]*10
for i in range(len(test_2000)):
    digito = test_2000[0][i]
    serie = test_2000.iloc[i][1:]
    df = serie.to_frame()
    vector_imagen = df.to_numpy()
    vector_imagen = vector_imagen.reshape((vector_imagen.shape[1],vector_imagen.shape[0]))
    lista_matrices_M[digito][lista_contadores[digito]] = vector_imagen
    lista_contadores[digito] = lista_contadores[digito] +  1
#%%%






