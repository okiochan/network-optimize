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
( Для поиска правой границе в спуске, воспользуемся золотым сечением )
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

**optimize** - Nonlinear conjugate gradient, принимает ф-ю которую минимизируем, ф-ю возвращающую градиент, кол-во итераций, остальное уже задано.
```
def optimize(f, g, x0, maxiter=2000, EPS=1e-6, verbose=True, printEvery=50):
```

Обучим сеть и запустим:
выведем : итерацию, норму градиента и значение функции (видим как значение ф-ии уменьшается =) )
И ошибку: 97% верно распознал

![](https://raw.githubusercontent.com/okiochan/network-optimize/master/img/i1.png)
 
 и сам результат классификации (сеть: 2 входных, 10 на скрытом, 1 на выходе)
 
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/img/i2.png)
 

# Оптимизация нейро сети

Алгоритм такой:

1) обучить на недостаточном количестве нейронов
2) после стабилизации весов, добавим новый нейрон (старые значения сохранены)
3) заново обучим сеть, при этом полезная информ, накопленная сетью, не потеряется

Добавляем нейрон в сеть так (у нас матрица - нам нужно добавить столбец и строку)
```
def AddNeuron(NN):
    a, b, c = NN.layers
    NN.layers[1] += 1
    NN.b[0] = np.concatenate((NN.b[0], [0]))
    n = NN.W[0].shape[1]
    NN.W[0] = np.hstack((NN.W[0], np.random.randn(a,1) * 1e-5))
    NN.W[1] = np.vstack((NN.W[1], np.random.randn(1,c) * 1e-5))
```

Посмотрим на примере

![](https://raw.githubusercontent.com/okiochan/network-optimize/master/img/i3.png)

картинка показывает как уменьшается ошибка при увеличении кол-ва нейронов для 1-го скрытого слоя. Для данного примера, 12 нейроновв достаточно

![](https://raw.githubusercontent.com/okiochan/network-optimize/master/img/i33.png)

На картинке : сколько нейронов взято, в каждой строке: кол-во итераций, норма градиента, значение функции

вот какая картинка получилась для 1-го скрытого слоя и 12 нейронов

![](https://raw.githubusercontent.com/okiochan/network-optimize/master/img/i4.png)

вот еще пример, 2 скрытых слоя по 10 и 5 нейронов

![](https://raw.githubusercontent.com/okiochan/network-optimize/master/img/i5.png)

# Optimal Brain Damage

![](https://raw.githubusercontent.com/okiochan/network-optimize/master/img/d1.png)

![](https://raw.githubusercontent.com/okiochan/network-optimize/master/img/d2.png)

Реализация salience

```
def Salience(NN, weights, X, Y):
    sd = SecondDerivatives(NN, weights, X, Y)
    res = np.zeros(weights.size)
    for i in range(weights.size):
        res[i] = weights[i] ** 2 * sd[i]
    return res
```

И поудаляем 5 раз веса(axons, Wi ) с наименьшими salience

```
for i in range(5):
    w_hat = conjugate_gradient.optimize(cost_wrapper, gradient_wrapper, NN.params(), maxiter=300)
    s = Salience(NN,w_hat, X, Y)
    useless = np.argmin(np.abs(s))
    w_hat[useless] = 0
    print("Removing axon {} with salience {}".format(useless, s[useless]))
    NN.params(w_hat)
```
Запустим программу, ошибка SSE стала в 2 раза меньше, чем в методе, описанном выше
Также мы видим какие аксоны удалялись

![](https://raw.githubusercontent.com/okiochan/network-optimize/master/img/d3.png)
 
Преимщества: В нейросети можно набрать много линейных функций и всего одну нелинейную и можно будет аппроксимировать любую функцию.
Используя способность обучения на множестве примеров, нейронная сеть способная решать задачи, в которых неизвестны закономерности развития ситуации и зависимости между входными и выходными данными. 