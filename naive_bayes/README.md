# Some mathematical theory

## Bayes

$P(A|B)=\frac{P(B|A)P(A)}{P(B)}$

The training data

$T=\{(x_1,y_1),(x_2, y_2),...,(x_N,y_N)\}$

And we can get the prior distribution

$P(Y=c_k),k=1,2,3...,K$

## The naive Bayes assumption

The contingent probability

$P(X=x|Y=c_k)=P(X^{(1)}=x^{(1)},...,X^{(n)}=x^{(n)}),k=1,2,...,K$

Get the joint probability $P(X,Y)$

If calculate $P(X,Y)$, there are too many parameter to estimate

So the naive Bayes made a bold assumption, that every variable is independent

In the assumption, $P(X,Y)$ can simply expression as
$$
\begin{align}
P(X=x|Y=c_k)&=P(X^{(1)}=x^{(1)},...,X^{(n)}=x^{(n)})\\
&=\prod^n_{j=1}P(X^{(j)}=x^{(j)}|Y=c_k)
\end{align}
$$

## Bayes classifier

Combine above mentioned formula, a final Bayes classifier can be summarize
$$
y=f(x)=argmax_{ck}{\frac{P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)}{\sum_kP(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)}}
$$
all the $c_k$ in the equation is same, so the equation can be simplify to 
$$
y=argmax_{c_k}P(Y=c_k)\prod_jP(X^{(j)}=x^{(j)}|Y=c_k)
$$

## Future update

about naive Bayes, I only realize a word corrector. And I'll update some new program with [three frequently used model](https://blog.csdn.net/qq_27009517/article/details/80044431)

## Reference

All the formula reference the 《统计学习方法》 and [This blog](https://blog.csdn.net/wyq_wyj/article/details/79485618)