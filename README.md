# 《概率论与数理统计》项目 2

- 杨凯
- 郭佑恒

这个项目主要就是填充[`projet.py`](./projet.py)文件，使得[`projet2-3i005.ipynb`](./projet2-3i005.ipynb)文件可以运行，并得到与[`projet2-3i005.pdf`](./projet2-3i005.pdf)同样的结果。

那一堆图片文件，不用管，是为了供`projet2-3i005.ipynb`这个 python 的 notebook 用的。

另外还有三个`.csv`数据集文件：

- `heart.csv`
- `train.csv`
- `test.csv`

具体的话在[`projet2-3i005.ipynb`](./projet2-3i005.ipynb)里有详细说明。

最后，有一个[`utils.py`](./utils.py)文件，里面有一些函数供我们使用：

### `getNthDict(df, n):`

- Rend un dictionnaire des données de la n-ième ligne

- Arguments:

  - df {pandas.dataframe} -- le pandas.dataframe contenant les données
  - n {int} -- le numéro de la ligne

### `viewData(data, kde=True):`

- visualisation d'un pandas.dataframe sous la forme d'histogramme (avec uu gaussian kernel densiy estimate si demandé et intéressant)

- Arguments:

  - data {pandas.dataFrame} -- le pandas.dataframe à visualiser

- Keyword Arguments:

  - kde {bool} -- demande de l'affichage du gaussian kdf (default: {True})

### `discretizeData(data):`

- Discrétisation automatique utilisant numpy

- Arguments:

  - data {pandas.dataframe} -- le dataframe dont certaines colonnes seront à discrétisées

- Returns:
  - pandas.dataframe -- le nouveau dataframe discréité

### `drawGraphHorizontal(arcs):`

- Dessine un graph (horizontalement) à partir d'une chaîne décrivant ses arcs (et noeuds) (par exemple 'A->B;C->A')"
- :param arcs: la chaîne contenant les arcs
- :return: l'image représentant le graphe

### `drawGraph(arcs):`

- Dessine un graph à partir d'une chaîne décrivant ses arcs (et noeuds) (par exemple 'A->B;C->A')"
- :param arcs: la chaîne contenant les arcs
- :return: l'image représentant le graphe

### 以及一个抽象类`class AbstractClassifier:`

- Un classifier implémente un algorithme pour estimer la classe d'un vecteur d'attributs. Il propose aussi comme service de calculer les statistiques de reconnaissance à partir d'un pandas.dataframe.
