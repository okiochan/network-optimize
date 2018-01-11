#non-linear

Реализован многослойный персептрон с **backpropagation**
Описание метода **bakpropagation** и выведенные рассчетные формулы лежат в [файле]( https://github.com/okiochan/network-optimize/blob/master/backprop.docx)

Обучала методом [Nonlinear conjugate gradient ]( https://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method)

Идея метода следующая:

Пусть есть функция 
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/n1.gif) переменных, которую нужно минимизировать. 

Обозначим
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/n2.gif)
градиенотом, дает нам направление вектора найскорейшего возрастания. 

Для минимизации, будем двигаться в направлении антиградиента:
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/n3.gif)

С шагом длины альфа, длину будем подбирать бин поиском.
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/n4.gif)

Во время первой итерации мы пойдем по направлению градиента
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/f1.gif)
 После каждой следующей итерации, направление будет происходить вдоль сопряженного направления ![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/f2.gif)
где 
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/f3.gif)

Выполним следующие шаги:

1) Посчитаем направление 
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/f4.gif)

2) Для расчета бетты, возьмем формулу Hestenes-Stiefel
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/f5.gif)

3) Обновим
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/f6.gif)

4) Оптимизируем направление
( Для поиска правой границе в спуске, воспользуемся [Ternary_search ]( https://en.wikipedia.org/wiki/Ternary_search) )
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/f7.gif)
 
5) обновим позицию ![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/f8.gif)

Для того, чтобы запустить сеть, нужно открыть **train.py**.

В 17й строке, в train.py, задается архитектура сети
Метод принимает список нейронов (он же и задает список слоев) и функции активации для каждого слоя
В файле **activations.py** есть много разных функций активаций, достаточно написать их имя в методе.

В данном примере у нас 3 слоя, 1 скрытый. Для 1го слоя функция активации не нужна - это слой входных данных
```
NN = nn.NeuralNetwork([2, 10, 1], ["tanh", "tanh"])
```

Данные нормализованы для тангенса, при смене ф-ии акт., эти строки нужно будет поменять
```
max_value = np.max(np.fabs(X))
X /= (max_value * 1.1)
Y = np.atleast_2d(Y).T
Y[Y == 0] = -1
```

В файле **nn.py** сама нейро сеть, ее основные методы **forward** (пропускаем вход к выходу), **__init__** - конструктор сети,
**cost** - SSE, **cheslennyi gradient** - был написан численный градиент, чтобы проверить backpropgation - **cost_grad**

В файле **conjugate_gradient.py** - обучение сети, **bins** - обычный бин поиск

**optimize** - Nonlinear conjugate gradient, принимает SSE, градиент, кол-во итераций, остальное уже задано.
```
def optimize(f, g, x0, maxiter=2000, gtol=1e-6, verbose=True, printEvery=50):
```

Обучим сеть и запустим:
выведем : итерацию, норму градиента и значение функции (видим как значение ф-ии уменьшается =) )
И ошибку: 97% верно распознал
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/img/i1.png)
 
 и сам результат классификации
 
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/img/i2.png)
 
