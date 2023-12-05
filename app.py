from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
from sklearn import metrics
import numpy as np
import graphviz

# Carregar a base de dados Iris
# X: features, y: rótulos
iris = load_iris()
Dado = iris.data
Tipo = iris.target

def Especificidade_Score(Tipo_True, Tipo_pred, Tipos):
    Especificidade_scores = []
    for Tipo in Tipos:
        True_Negativo = np.sum((Tipo_True != Tipo) & (Tipo_pred != Tipo))
        False_Positivo = np.sum((Tipo_True != Tipo) & (Tipo_pred == Tipo))
        Especificidade = True_Negativo / (True_Negativo + False_Positivo) if (True_Negativo + False_Positivo) != 0 else 0
        Especificidade_scores.append(Especificidade)
    return np.mean(Especificidade_scores)


# Dividir os dados em treinamento e teste (A, B e C)
Dado_A_B, Dado_C, Tipo_A_B, Tipo_C = train_test_split(Dado, Tipo, test_size=1/3, stratify=Tipo, random_state=42)
Dado_A, Dado_B, Tipo_A, Tipo_B = train_test_split(Dado_A_B, Tipo_A_B, test_size=1/2, stratify=Tipo_A_B, random_state=42)

# Número de experimentos
N = 3

# Inicializar listas para armazenar métricas
Acuracia = []
Sensitividade = []
Especificidade = []
Precisao = []

for i in range(N):
    if i == 0:
        Dado_Treino, Dado_Teste, Tipo_Treino, Tipo_Teste = Dado_A_B, Dado_C, Tipo_A_B, Tipo_C
    elif i == 1:
        Dado_Treino, Dado_Teste, Tipo_Treino, Tipo_Teste = np.concatenate((Dado_A, Dado_C)), Dado_B, np.concatenate((Tipo_A, Tipo_C)), Tipo_B
    else:
        Dado_Treino, Dado_Teste, Tipo_Treino, Tipo_Teste = np.concatenate((Dado_C, Dado_B)), Dado_A, np.concatenate((Tipo_C, Tipo_B)), Tipo_A

    # Árvores de Decisão
    AD_Classificacao = DecisionTreeClassifier(criterion='entropy')
    AD_Classificacao.fit(Dado_Treino, Tipo_Treino) # treina
    AD_pred = AD_Classificacao.predict(Dado_Teste) # testa

    # KNN
    KNN_Classificacao = KNeighborsClassifier()
    KNN_Classificacao.fit(Dado_Treino, Tipo_Treino)
    KNN_pred = KNN_Classificacao.predict(Dado_Teste)

    # Calcular métricas para Árvores de Decisão
    AD_Acuracia = metrics.accuracy_score(Tipo_Teste, AD_pred)
    AD_Sensitividade = metrics.recall_score(Tipo_Teste, AD_pred, average='weighted')
    AD_Precisao = metrics.precision_score(Tipo_Teste, AD_pred, average='weighted')
    AD_Especificidade = Especificidade_Score(Tipo_Teste, AD_pred, Tipos=[0, 1, 2])

    # Calcular métricas para KNN
    KNN_Acuracia = metrics.accuracy_score(Tipo_Teste, KNN_pred)
    KNN_Sensitividade = metrics.recall_score(Tipo_Teste, KNN_pred, average='weighted')
    KNN_Precisao = metrics.precision_score(Tipo_Teste, KNN_pred, average='weighted')
    KNN_Especificidade = Especificidade_Score(Tipo_Teste, KNN_pred, Tipos=[0, 1, 2])

    # Armazenar métricas
    Acuracia.append([AD_Acuracia, KNN_Acuracia])
    Sensitividade.append([AD_Sensitividade, KNN_Sensitividade])
    Precisao.append([AD_Precisao, KNN_Precisao])
    Especificidade.append([AD_Especificidade, KNN_Especificidade])

    tree_structure = AD_Classificacao.tree_
    dot_data = export_graphviz(AD_Classificacao, out_file=None, 
                           feature_names=iris.feature_names,  
                           class_names=iris.target_names,  
                           filled=True, rounded=True,  
                           special_characters=True)
    # Visualizar a árvore
    graph = graphviz.Source(dot_data)  
    graph.render("arvore_decisao_"+str(i+1))
    graph.view("arvore_decisao_"+str(i+1))


    print("\nMétricas do teste " + str(i+1)+ ":")
    print(f"Acurácia: {[AD_Acuracia, KNN_Acuracia]}")
    print(f"Sensitividade: {[AD_Sensitividade, KNN_Sensitividade]}")
    print(f"Especificidade: {[AD_Precisao, KNN_Precisao]}")
    print(f"Precisão: {[AD_Especificidade, KNN_Especificidade]}")

# Calcular média das métricas
Media_Acuracia = np.mean(Acuracia, axis=0)
Media_Sensitividade = np.mean(Sensitividade, axis=0)
Media_Precisao = np.mean(Precisao, axis=0)
Media_Especificidade = np.mean(Especificidade, axis=0)

# Apresentar resultados médios
print("\nMédias das Métricas:")
print(f"Acurácia: {Media_Acuracia}")
print(f"Sensitividade: {Media_Sensitividade}")
print(f"Especificidade: {Media_Especificidade}")
print(f"Precisão: {Media_Precisao}")