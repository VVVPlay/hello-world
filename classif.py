import numpy as np
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Для управления 3D-изображением (посредством мышки)
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import sys  # Для sys.exit()

np.random.seed(348)
# Загрузка данных
iris = pandas.read_csv('wine.csv')
print(iris.shape)  # (150, 5)
print(iris.head(10))  ## Печать первых 10 строк данных, загруженных из файла iris.csv
print(iris.describe())  # count, mean, std и др.
print(iris.groupby('quality').size())

# Ветви программы
hist = False  # True - вывод гистограмм
groundTruth = True  # True - вывод классов по трем характеристикам: ширина лепестка, длина чашелистика, длина лепестка
cross_val = False  # True, False - вывод карт вероятности по двум характеристикам: длина и ширина чашелистика
#
if hist:
    ## Гистограммы характеристик ирисов
    iris.hist()
    plt.show()
    sys.exit()
if groundTruth:
    # Заменяем имена классов на номера
    iris["color"] = iris["color"].map({"white": 0, "red": 1})
    # Выводим результаты наблюдений (по 3-м характеристикам)
    X = iris.values[:, 0:4]
    y = iris.values[:, 4]
    fig = plt.figure(1, figsize=(5, 4))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    for name, label in [('white', 0), ('red', 1)]:
        ax.text3D(X[y == label, 3].mean(),
                  X[y == label, 0].mean(),
                  X[y == label, 2].mean() + 2, name,
                  horizontalalignment='center',
                  bbox=dict(alpha=0.2, edgecolor='w', facecolor='w'))
    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    '''ax.set_xlabel('Ширина лепестка')
    ax.set_ylabel('Длина чашелистика')
    ax.set_zlabel('Длина лепестка')
    ax.set_title('Ирисы Фишера')'''
    ax.dist = 12
    fig.show()
    sys.exit()
iris_array = iris.values
x = iris_array[:, 0:4]
y = iris_array[:, 4]
classifiers = []
classifiers.append(('RF', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))
# Оценка методов
results = []
names = []
k = 10
for name, classifier in classifiers:
    # Генерируем индексы для выделения обучающих и тестовых данных
    # k - 1 подвыборок будут использованы для обучения, а одна - для проверки (валидации)
    kfold = model_selection.KFold(n_splits=k, random_state=348, shuffle=True)
    ## cv - стратегия кросс-валидации
    cv_results = model_selection.cross_val_score(classifier, x, y, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    # print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))
    print(name, np.round(cv_results, 3), 'Средняя точность: ', round(cv_results.mean(), 3))
# Графическое представление результатов
fig = plt.figure()
fig.suptitle('Точность, показанная классификаторами')
ax = fig.add_subplot(1, 1, 1)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
