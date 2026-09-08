# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


X_train = "/Users/ramiro/Desktop/TP1/X_train.csv"
y_train = "/Users/ramiro/Desktop/TP1/y_train.csv"
X_tests = "/Users/ramiro/Desktop/TP1/X_test.csv"

# Armado de los DataFrames:

predictores = pd.read_csv(X_train)
var_ind = pd.read_csv(y_train)["weight"]
X_test = pd.read_csv(X_tests)

print("--- Predictores:")
print(predictores)
print("\n----------------------------------------------------------------------------------------------------------------------------------")
print("\n--- Variable independiente:")
print(var_ind)
print("\n==================================================================================================================================")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("\n--- Exploración de los datos. Distribución del peso de los bebés al nacer")
print("\n--- Estadísticos:")

media = np.mean(var_ind)
varianza = np.var(var_ind)
desviacion = np.std(var_ind)
mediana = np.median(var_ind)

print("- Media:", media)
print("- Mediana:", mediana)
print("- Varianza:", varianza)
print("- Desviación estándar:", desviacion)

print("\n--- Verificación de distribución normal por los tests de Kolmogorov y Shapiro-Wilk. Se encarga a ChatGPT que escriba los \nrespectivos códigos en Python.")

resultado = stats.kstest(
    var_ind,
    'norm',
    args=(np.mean(var_ind), np.std(var_ind))
)

print("- Estadístico K-S:", resultado.statistic)
print("- p-valor:", resultado.pvalue)

# Shapiro-Wilk
resultado = stats.shapiro(var_ind)

print("\n- Estadístico Shapiro-Wilk:", resultado.statistic)
print("- p-valor:", resultado.pvalue)

print ("\nEn ambos tests, el p-value resulta < 0,05, por lo que se rechaza la hipótesis nula que plantea que los datos siguen una \ndistribución normal.")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


print("\n--- Histograma:")

plt.hist(var_ind, bins=70, range=[0.5, 14], rwidth=0.85)

plt.xlabel("Peso del bebé al nacer (libras)")
plt.ylabel("Frecuencia")
plt.title("Distribución del peso de neonatos")

plt.show()

print("\nVisualmente, el histograma se aproxima a una distribución normal. Es bastante simétrico, algo que también se interpreta al comparar \nla media con la mediana.")

print("\n==================================================================================================================================")
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("\n--- Análsis de distribuciones de los predictores:")

print("\n-- Predictores numéricos: fage, mage, visits y gained")

print("\n- Caracterización:")

print(predictores[["fage", "mage", "visits", "gained"]].describe())

predictores[["fage", "mage", "visits", "gained"]].hist(
    bins=20,
    figsize=(10, 8)
)

plt.show()

print("\n-- Análisis de normalidad (test de Shapiro-Wilk):")

resultado_fage = stats.shapiro(predictores["fage"])
resultado_mage = stats.shapiro(predictores["mage"])
resultado_visits = stats.shapiro(predictores["visits"])
resultado_gained = stats.shapiro(predictores["gained"])

print("- fage:", resultado_fage)
print("- mage:", resultado_mage)
print("- visits:", resultado_visits)
print("- gained:", resultado_gained)

print("\n--- Ninguno de los predictores (numéricos) sigue una distribución normal (p < 0,05). Sin embargo, tanto de los gráficos como de la \ncomparación de las medias y medianas surge que estas variables siguen distribuciones bastante simétricas.") 

print("\n----------------------------------------------------------------------------------------------------------------------------------")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("\n-- Predictores categóricos: mature, sex, habit, marital y whitemom")

variables_cat = ["mature", "sex", "habit", "marital", "whitemom"]

for variable in variables_cat:
    #print(variable)
    print(predictores[variable].value_counts())
    print()



print("\n=================================================================================================================================")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("\n--- Análisis de la distribución de la variable independiente entre cada uno de los predictores categóricos: \nSe pide a ChatGPT que escriba el código en Python para la clasificación y obtención de parámetros y análisis estadístico \n(t-test). Importante: para los tests estadístico se asumió homocedasticidad en todos los casos.")


print("\n-- Influencia del sexo del bebé en su peso al nacer:")

plt.boxplot(
    [
        var_ind[predictores["sex"] == "female"],
        var_ind[predictores["sex"] == "male"]
    ],
    labels=["Female", "Male"]
)

plt.xlabel("Sexo")
plt.ylabel("Peso del bebé al nacer (libras)")
plt.title("Distribución del peso al nacer según sexo")

plt.show()

print("Peso según sexo:")

print("\nFemale:")
print(var_ind[predictores["sex"] == "female"].describe())

print("\nMale:")
print(var_ind[predictores["sex"] == "male"].describe())

print("Se ven diferencias en las medias y medianas entre los distintos sexos. Se pide a ChatGPT el código para verificarlo por t-test:")

resultado_ttest = stats.ttest_ind(
    var_ind[predictores["sex"] == "female"],
    var_ind[predictores["sex"] == "male"]
)

print("- Estadístico t:", resultado_ttest.statistic)
print("- p-value:", resultado_ttest.pvalue)

print("\nEfectivamente, el peso de los varones es significativamente mayor al de las mujeres en esta muestra al momento de nacer \n(p < 0.001).")

print("\n--------------------")

print("\n-- Influencia de la edad de la madre en el peso de bebé al nacer:")

plt.boxplot(
    [
        var_ind[predictores["mature"] == "younger mom"],
        var_ind[predictores["mature"] == "mature mom"]
    ],
    labels=["Younger mom", "Mature mom"]
)

plt.xlabel("Madurez materna")
plt.ylabel("Peso del bebé al nacer (libras)")
plt.title("Distribución del peso al nacer según madurez materna")

plt.show()

print("Peso según madurez materna:")

print("\nYounger mom:")
print(var_ind[predictores["mature"] == "younger mom"].describe())

print("\nMature mom:")
print(var_ind[predictores["mature"] == "mature mom"].describe())

resultado_ttest = stats.ttest_ind(
    var_ind[predictores["mature"] == "younger mom"],
    var_ind[predictores["mature"] == "mature mom"]
)

print("- Estadístico t:", resultado_ttest.statistic)
print("- p-value:", resultado_ttest.pvalue)

print("\nSe encontraron diferencias estadísticamente significativas en el peso al nacer según la edad materna (p < 0,05). \nLos hijos de madres de 35 años o más presentaron un mayor peso promedio que los hijos de madres menores de 35 años.")

print("\n--------------------") 

print("\n-- Distribución del peso según hábito de fumar:")

plt.boxplot(
    [
        var_ind[predictores["habit"] == "nonsmoker"],
        var_ind[predictores["habit"] == "smoker"]
    ],
    labels=["Nonsmoker", "Smoker"]
)

plt.xlabel("Hábito de fumar")
plt.ylabel("Peso del bebé al nacer (libras)")
plt.title("Distribución del peso al nacer según hábito de fumar")

plt.show()

print("Peso según hábito de fumar:")

print("\nNonsmoker:")
print(var_ind[predictores["habit"] == "nonsmoker"].describe())

print("\nSmoker:")
print(var_ind[predictores["habit"] == "smoker"].describe())

resultado_ttest = stats.ttest_ind(
    var_ind[predictores["habit"] == "nonsmoker"],
    var_ind[predictores["habit"] == "smoker"]
)

print("- Estadístico t:", resultado_ttest.statistic)
print("- p-value:", resultado_ttest.pvalue)

print("\nNuevamente, el hecho de que la madre fume afecta el peso del bebé al nacer (p < 0.001). En este sentio, los bebés de madres \nfumadoras presentaron un peso promedio significativamente menor respecto de de aquellos de madres no fumadoras.")

print("\n--------------------")

print("\n-- Influencia del estado civil de los progenitores en el peso del neonato:")

plt.boxplot(
    [
        var_ind[predictores["marital"] == "married"],
        var_ind[predictores["marital"] == "not married"]
    ],
    tick_labels=["Married", "Not married"]
)

plt.xlabel("Estado civil")
plt.ylabel("Peso del bebé al nacer (libras)")
plt.title("Distribución del peso al nacer según estado civil")

plt.show()

print("Peso según estado civil:")

print("\nMarried:")
print(var_ind[predictores["marital"] == "married"].describe())

print("\nNot married:")
print(var_ind[predictores["marital"] == "not married"].describe())

resultado_ttest = stats.ttest_ind(
    var_ind[predictores["marital"] == "married"],
    var_ind[predictores["marital"] == "not married"]
)

print("- Estadístico t:", resultado_ttest.statistic)
print("- p-value:", resultado_ttest.pvalue)

print("\nEl estado marital de la madre no afectaría el peso del bebé al nacer.")

print("\n--------------------")


print("\nDistribución del peso según 'raza' de la madre:")

plt.boxplot(
    [
        var_ind[predictores["whitemom"] == "white"],
        var_ind[predictores["whitemom"] == "not white"]
    ],
    tick_labels=["White", "Not white"]
)

plt.xlabel("Raza materna")
plt.ylabel("Peso del bebé al nacer (libras)")
plt.title("Distribución del peso al nacer según raza materna")

plt.show()

print("Peso según raza materna:")

print("\nWhite:")
print(var_ind[predictores["whitemom"] == "white"].describe())

print("\nNot white:")
print(var_ind[predictores["whitemom"] == "not white"].describe())

resultado_ttest = stats.ttest_ind(
    var_ind[predictores["whitemom"] == "white"],
    var_ind[predictores["whitemom"] == "not white"]
)

print("- Estadístico t:", resultado_ttest.statistic)
print("- p-value:", resultado_ttest.pvalue)

print("\nSe encontraron diferencias estadísticamente significativas en el peso al nacer según la categoría de raza materna \nLos hijos de madres clasificadas como white presentaron un peso promedio mayor que aquellos de madres clasificadas \ncomo not white (p < 0,001).")

print("\n--------------------")

print("\n--- En resumen:")
print("Los predictores categóricos sex, mature, habit and whitemom influyen significativamente en el peso del bebé (p < 0,05). No hay \nevidencia suficiente para afirmar que el estado marital también lo haga.")


print("\n=================================================================================================================================")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("\n--- Análisis de correlación entre los predictotes (individualmente) y el peso de los bebés al nacer:")

print("\nEdad del padre y el peso de su hijo al nacer:")


plt.scatter(predictores["fage"], var_ind)

plt.xlabel("Edad del padre (años)")
plt.ylabel("Peso del bebé al nacer (libras)")
plt.title("Relación entre edad paterna y peso al nacer")

print("\nCorrelación de Pearson:")

plt.show()

resultado = stats.pearsonr(predictores["fage"], var_ind)

print("\n- Coeficiente de correlación:", resultado.statistic)
print("- p-value:", resultado.pvalue)

print("\nLa correlación fue positiva, pero de muy baja magnitud. El p-value indica que no hay correlación lineal \nestadísticamente significativa.")

print("\n--------------------")

print("\nEdad de la madre y el peso de su hijo al nacer:")

plt.scatter(predictores["mage"], var_ind)

plt.xlabel("Edad de la madre (años)")
plt.ylabel("Peso del bebé al nacer (libras)")
plt.title("Relación entre edad materna y peso al nacer")

plt.show()

print("\nCorrelación de Pearson:")

resultado = stats.pearsonr(predictores["mage"], var_ind)

print("\n- Coeficiente de correlación:", resultado.statistic)
print("- p-value:", resultado.pvalue)

print("Existe una correlación positiva y estadísticamente significativa entre la edad materna y el peso promedio \ndel bebé al nacer (p < 0,01). Sin embargo, la magnitud de la correlación fue baja, lo que indica una asociación lineal débil.")


print("\n--------------------")

print("\nRelación entre el número de visitas al obstetra y el peso de su hijo al nacer:")

plt.scatter(predictores["visits"], var_ind)

plt.xlabel("Cantidad de visitas al obstetra")
plt.ylabel("Peso del bebé al nacer (libras)")
plt.title("Relación entre la cantidad de visitas al obstetra y peso al nacer")

plt.show()

print("\nCorrelación de Pearson:")

resultado = stats.pearsonr(predictores["visits"], var_ind)

print("\n- Coeficiente de correlación:", resultado.statistic)
print("- p-value:", resultado.pvalue)

print("\nNo se observó una correlación lineal estadísticamente significativa entre la cantidad de visitas al obstetra y el peso \nal nacer. La correlación es positiva, pero de muy baja magnitud.")

print("\n--------------------")

print("\nCorrelación entre el peso corporal ganado por la madre y el peso de su hijo al nacer:")

plt.scatter(predictores["gained"], var_ind)

plt.xlabel("Incrememnto de peso corporal durante el embarazo")
plt.ylabel("Peso del bebé al nacer (libras)")
plt.title("Relación entre el peso corporal ganado por la madre durante el embarazo y peso al nacer")

plt.show()

print("\nCorrelación de Pearson:")

resultado = stats.pearsonr(predictores["gained"], var_ind)

print("\n- Coeficiente de correlación:", resultado.statistic)
print("- p-value:", resultado.pvalue)

print("\nSe observó una correlación positiva y estadísticamente significativa entre el aumento de peso materno y el peso \nal nacer (p < 0,01). No obstante, la magnitud de la correlación fue muy baja, indicando una asociación lineal débil entre ambas \nvariables.")

print("\n=================================================================================================================================")

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

print("\nCONSTRUCCIÓN DEL MODELO")

print("\nPrimeramente, las variables categóricas se convierten en binarias y se agregan como nuevas series al DataFrame de predictores.")


predictores["mature_bin"] = predictores["mature"].map({
    "younger mom": 0,
    "mature mom": 1
})

predictores["sex_bin"] = predictores["sex"].map({
    "female": 0,
    "male": 1
})

predictores["habit_bin"] = predictores["habit"].map({
    "nonsmoker": 0,
    "smoker": 1
})

predictores["marital_bin"] = predictores["marital"].map({
    "not married": 0,
    "married": 1
})

predictores["whitemom_bin"] = predictores["whitemom"].map({
    "not white": 0,
    "white": 1
})


print("Luego, se dividen (aleatoriamente) los datos de modo de usar una parte (mayoritaria) para el entrenamiento -plantelo del modelo-, \ny los restantes para la validación de la regresión lograda.")

X_entrenamiento, X_validación, y_entrenamiento, y_validación = train_test_split(
    predictores,
    var_ind,
    test_size=0.20,
    random_state=42
)

print("\nTamaño de X_entrenamiento:", X_entrenamiento.shape)
print("Tamaño de X_validación:", X_validación.shape)

print("\nTamaño de y_entrenamiento:", y_entrenamiento.shape)
print("Tamaño de y_validación:", y_validación.shape)

print("\n------------------------------------------------------------------------------------------------------------------------------")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

print("\nMODELOS:")

print("\nModelo 1:")

print("\nModelo lineal con predictores: fage, mage, visits, gained,sex_bin, habit_bin, marital_bin, whitemom_bin, mature_bin")

features1 = ["fage", "mage", "visits", "gained", "sex_bin", "habit_bin", "marital_bin", "whitemom_bin", "mature_bin"]

X = np.array(X_entrenamiento[features1])
y = np.array(y_entrenamiento)
reg = LinearRegression()
reg.fit(X, y)

MAE1_train = mean_absolute_error(y, reg.predict(X))
R21_train = r2_score(y, reg.predict(X))

y_pred_val = reg.predict(np.array(X_validación[features1]))
MAE1 = mean_absolute_error(y_validación, y_pred_val)
R21 = r2_score(y_validación, y_pred_val)

print("\nMAE entrenamiento:", MAE1_train, "| MAE validación:", MAE1)
print("R² entrenamiento:", R21_train, "| R² validación:", R21)

print("\n--------------------")

print("\nModelo 2:")

print("\nModelo lineal con predictores: mage, visits, gained,sex_bin, habit_bin, whitemom_bin")


features2 = ["mage", "visits", "gained", "sex_bin", "habit_bin", "whitemom_bin"]

X = np.array(X_entrenamiento[features2])
y = np.array(y_entrenamiento)
reg = LinearRegression()
reg.fit(X, y)

MAE2_train = mean_absolute_error(y, reg.predict(X))
R22_train = r2_score(y, reg.predict(X))

y_pred_val = reg.predict(np.array(X_validación[features2]))
MAE2 = mean_absolute_error(y_validación, y_pred_val)
R22 = r2_score(y_validación, y_pred_val)

print("\nMAE entrenamiento:", MAE2_train, "| MAE validación:", MAE2)
print("R² entrenamiento:", R22_train, "| R² validación:", R22)

print("\n--------------------")

print("\nModelo 3:")

print("\nModelo lineal con predictores: mage, gained,habit_bin, whitemom_bin")

features3 = ["mage", "gained", "habit_bin", "whitemom_bin"]

X = np.array(X_entrenamiento[features3])
y = np.array(y_entrenamiento)
reg = LinearRegression()
reg.fit(X, y)

MAE3_train = mean_absolute_error(y, reg.predict(X))
R23_train = r2_score(y, reg.predict(X))

y_pred_val = reg.predict(np.array(X_validación[features3]))
MAE3 = mean_absolute_error(y_validación, y_pred_val)
R23 = r2_score(y_validación, y_pred_val)

print("\nMAE entrenamiento:", MAE3_train, "| MAE validación:", MAE3)
print("R² entrenamiento:", R23_train, "| R² validación:", R23)

print("\n--------------------")

print("\nModelo 4:")

print("\nModelo lineal con predictores: mage, gained,whitemom_bin")

features4 = ["mage", "gained", "whitemom_bin"]

X = np.array(X_entrenamiento[features4])
y = np.array(y_entrenamiento)
reg = LinearRegression()
reg.fit(X, y)

MAE4_train = mean_absolute_error(y, reg.predict(X))
R24_train = r2_score(y, reg.predict(X))

y_pred_val = reg.predict(np.array(X_validación[features4]))
MAE4 = mean_absolute_error(y_validación, y_pred_val)
R24 = r2_score(y_validación, y_pred_val)

print("\nMAE entrenamiento:", MAE4_train, "| MAE validación:", MAE4)
print("R² entrenamiento:", R24_train, "| R² validación:", R24)


print("\n--------------------")

print("\nModelo 5:")

print("\nModelo lineal con predictores (features combinadas): visits/mage, gained, gained/visits, sex_bin*gained, habit_bin*gained, visits*mage")

features5 = ["visits/mage", "gained", "gained/visits", "sex_bin*gained", "habit_bin*gained", "visits*mage"]

visitas_seguras = X_entrenamiento["visits"].replace(0.0, 1.0)

X_entrenamiento["visits/mage"] = X_entrenamiento["visits"] / X_entrenamiento["mage"]
X_entrenamiento["gained/mage"] = X_entrenamiento["gained"] / X_entrenamiento["mage"]
X_entrenamiento["gained/visits"] = X_entrenamiento["gained"] /visitas_seguras
X_entrenamiento["habit_bin*gained"] = X_entrenamiento["habit_bin"] * X_entrenamiento["gained"]
X_entrenamiento["sex_bin*gained"] = X_entrenamiento["sex_bin"] * X_entrenamiento["gained"]
X_entrenamiento["visits*mage"] = X_entrenamiento["visits"] * X_entrenamiento["mage"]

visitas_seguras_val = X_validación["visits"].replace(0.0, 1.0)

X_validación["visits/mage"] = X_validación["visits"] / X_validación["mage"]
X_validación["gained/visits"] = X_validación["gained"] /visitas_seguras_val
X_validación["habit_bin*gained"] = X_validación["habit_bin"] * X_validación["gained"]
X_validación["sex_bin*gained"] = X_validación["sex_bin"] * X_validación["gained"]
X_validación["visits*mage"] = X_validación["visits"] * X_validación["mage"]

X = np.array(X_entrenamiento[features5])
y = np.array(y_entrenamiento)
reg = LinearRegression()
reg.fit(X, y)

MAE5_train = mean_absolute_error(y, reg.predict(X))
R25_train = r2_score(y, reg.predict(X))

y_pred_val = reg.predict(np.array(X_validación[features5]))
MAE5 = mean_absolute_error(y_validación, y_pred_val)
R25 = r2_score(y_validación, y_pred_val)

print("\nMAE entrenamiento:", MAE5_train, "| MAE validación:", MAE5)
print("R² entrenamiento:", R25_train, "| R² validación:", R25)

print("\n--------------------")

print("\nModelo 6:")

print("\nModelo lineal con predictores: mage, gained,habit_bin, whitemom_bin (excluyendo pesos extremos: weight < 4 libras)")

features6 = ["mage", "gained", "habit_bin", "whitemom_bin"]

mascara_entrenamiento = y_entrenamiento >= 4
mascara_validación = y_validación >= 4

X6_entrenamiento = X_entrenamiento.loc[mascara_entrenamiento, features6]
y6_entrenamiento = y_entrenamiento[mascara_entrenamiento]
X6_validación = X_validación.loc[mascara_validación, features6]
y6_validación = y_validación[mascara_validación]

print("\nSe excluyen", (~mascara_entrenamiento).sum(), "casos en entrenamiento y", (~mascara_validación).sum(), "casos en validación con peso < 4 libras")

X = np.array(X6_entrenamiento)
y = np.array(y6_entrenamiento)
reg = LinearRegression()
reg.fit(X, y)

MAE6_train = mean_absolute_error(y, reg.predict(X))
R26_train = r2_score(y, reg.predict(X))

y_pred_val = reg.predict(np.array(X6_validación))
MAE6 = mean_absolute_error(y6_validación, y_pred_val)
R26 = r2_score(y6_validación, y_pred_val)

print("\nMAE entrenamiento:", MAE6_train, "| MAE validación:", MAE6)
print("R² entrenamiento:", R26_train, "| R² validación:", R26)

print("\n--------------------")

print("\nCOMPARACIÓN DE MODELOS (sobre datos de validación):")
print("Modelo 1 -> MAE:", MAE1, "| R²:", R21)
print("Modelo 2 -> MAE:", MAE2, "| R²:", R22)
print("Modelo 3 -> MAE:", MAE3, "| R²:", R23)
print("Modelo 4 -> MAE:", MAE4, "| R²:", R24)
print("Modelo 5 -> MAE:", MAE5, "| R²:", R25)
print("Modelo 6 -> MAE:", MAE6, "| R²:", R26, "(sin pesos < 4 libras)")






# %%