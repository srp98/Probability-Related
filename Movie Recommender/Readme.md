# Movie Recommendations

A simple program to generate movie recommendations using bayesian inference of random variables

## Requirements
- Python 3.8+
- Numpy
- Scipy
- Matplotlib

## Dataset
The `Data` folder contains files related to movie names, movie id's and user ratings of movies they have seen

## Problem Statement
Let $X\in \{ 0,\ldots ,M-1\}$ be a discrete random variable that represents the true rating (i.e. quality) of a movie you are considering, where 0 means awful, and $ M - 1$ 1 means amazing (you could think of these as the number of stars). X is distributed according to $p_{X}(\cdot )$.

Let $Y_ n\in \{ 0,\ldots ,M-1\}$ be a discrete random variable that represents the rating user n provided for this movie, where $n=0, \dots , N-1.$ User ratings are noisy. We will assume that conditioned on the movie's true/inherent rating $X$, the ratings of different users are independent and identically distributed such that $p_{Y_ n\mid X}(y \mid x) = p_{Y \mid X}(y \mid x)$ for $n=0,1,\ldots ,N-1.$ In other words, the joint probability distribution for $X, Y_0, Y_1, \dots, Y_{N-1}$ has the factorization


$$p_{X,Y_0,Y_1,\dots ,Y_{N-1}}(x,y_0,y_1,\dots ,y_{N-1}) = p_ X(x) \prod _{n=0}^{N-1} p_{Y|X}(y_ n \mid x).$$

So we apply bayesian rule to get $p_{X \mid Y_0,\dots ,Y_{N-1}}(x\mid y_0,\dots ,y_{N-1})$ in terms of the given functions $p_{X}(\cdot )$ and $p_{Y \mid X}(\cdot \mid \cdot )$.

We have:

$$\begin{align}
p_{X\mid Y_0,\dots ,Y_{N-1}}(x\mid y_0,\dots ,y_{N-1}) &= \frac{p_{Y_0,\dots , Y_{N-1} \mid X}(y_0,\dots ,y_{N-1}\mid x) p_{X}(x)}{p_{Y_0,\dots ,Y_{N-1}}(y_0,\dots ,y_{N-1})} \\
&= \frac{\prod _{n=0}^{N-1} p_{Y_ n \mid X}(y_ n\mid x) p_{X}(x)}{p_{Y_0,\dots ,Y_{N-1}}(y_0,\dots ,y_{N-1})}
\end{align}$$

where the first step is just Bayes' theorem, and the second step comes from conditional independence.

Now, using the theorem of total probability, we have in the denominator,

$$\begin{align}
p_{Y_0,\dots ,Y_{N-1}}(y_0,\dots ,y_{N-1}) &= \sum _{m = 0}^{M-1} p_{Y_0,\dots ,Y_{N-1},X}(y_0,\dots ,y_{N-1},m) \\

&= \sum _{m=0}^{M-1} p_{Y_0,\dots ,Y_{N-1}\mid X}(y_0,\dots ,y_{N-1}\mid m)p_{X}(m) \\
&= \sum _{m=0}^{M-1} p_{X}(m) \prod _{n=0}^{N-1} p_{Y_ n\mid X}(y_ n\mid m) \\
\end{align}$$

Finally, we use the fact that the $Y_ ns$ are identically distributed conditioned upon $X$, so that $p_{Y_ n\mid X}(y\mid x) = p_{Y\mid X}(y\mid x)$. Putting this together with the above two equations, we obtain

$$ p_{X\mid Y_0,\dots,Y_{N-1}}(x\mid y_0,\dots,y_{N-1}) = \boxed{\frac{p_{X}(x) \prod_{n=0}^{N-1} p_{Y\mid X}(y_n\mid x) }{\sum_{m = 0}^{M-1} p_{X}(m) \prod_{n=0}^{N-1} p_{Y\mid X}(y_n\mid m)}}
