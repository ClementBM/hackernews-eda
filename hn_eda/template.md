# Text item

We define a text item, can be any kind of text object, like a title, a scope or a paragraph for example.
This list of text items is noted, with $ 1 < i < N $

$$
\mathcal{O} = \{ o_i \}
$$

We define a list of all tokens, with $ 1 < i < N $ and $ 1 < j < M_i $, with $M_i$ the number of tokens of the object $i$.

$$
\mathcal{T} = \{ t_{ij} \}
$$

We define a segmentation/tokenization function $f_t(.)$, which associate to the text item $o_i$ a list of tokens $t_j$ such as
$$
f_t: o_i \rightarrow [t_{i1}, ..., t_{iM_i}]
$$

with $M_i$ the number of tokens of the object $i$

We define a dictionary $\mathcal{D} = \{ t_i \}$ that has only unique tokens