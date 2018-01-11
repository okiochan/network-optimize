#non-linear

Реализован многослойный персептрон с **backpropagation**
Описание метода **bakpropagation** лежит в [файле]( https://github.com/okiochan/network-optimize/blob/master/backprop.docx)

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

4) Бин поиском найдем альфу
![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/f7.gif)
 
5) обновим позицию ![](https://raw.githubusercontent.com/okiochan/network-optimize/master/formula/f8.gif)
 
 
 
 
 
