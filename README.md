# Multi-label
**author: Xiaowu He. horace_hxw_cal@berkeley.edu**

the data is availble here: **[The Extreme Classification Repository: Multi-label Datasets & Code](http://manikvarma.org/downloads/XC/XMLRepository.html)**

Just download one and put it in a /data directory

## 1. ECOC(multi-class setting)
#### step 1: map the class to the coresponding **binary** vector $[1,0,...,1]^T$ using the error correction codes.:
- Exhaustive code
- Col Selection from Exhaustive code (Hill Climbing)
- Randomize Hill Climbing
- BCH code

#### step 2: use maching learning algorithms to create the probability vector of each position to be 1.
- Decision Tree
- Nueral Network
- **create muliple binary classifiers for each position**

#### step 3： recover the origin class vector according to the cloest $\ell_1$ distance

## 2. MLGT (Group Testing)
The basic idea is to form group of labels and test each data point whether or not inside that group.
1. Construct  a specific binary matrix $A_{m\times d}$ and compress the original binary-label vector by $z = A \quad or \quad  y$ using matrix boolean opration. Then train the classifiers based on vector z.
2. Predicting process is to set the position $l$ of group vector to 1 iff $|supp(A^{(l)}) - supp(\hat{z}))| < \frac{e}{2}$. where the supp() means the set of indexees of nonzero enties in a given vector.

## 3. Haffman Tree Based multi-class Classification
constructing a haffman tree based on the frequency of each label. and train a binary classifier at each node of the tree.

**this can reduce the average classification and training time.**

## 4. ECOC in Multi-label setting
Assume there are m labels in total and each data point has no more than k labels.
1. we can map each label into a binary representation and concat them into a binary vector of leangth $k\log m$. If some data point doesn't have enough labels, add a default number to maintain the length.
2. Add **parity check**`    ` to the above vector using Error Correction Code
2. Train binary classifier on each digit.

## 5. Binary mapping + kNN
data set $(x,y)^d$, where $y_i=\{1, 0\}^L$

we want to map y into lower space by $$z = [M\cdot y]$$ where M is a multivariant i,i,d Gaussian matrix, and $[]$ is tkaing the sign.

Then we train binary classifiers on each bit of $z \in \{0, 1\}^{\hat L}$

For each test point, we predict its $\hat z$ and then use kNN to find the nearest k neighbors from $z=[My]$ which is all our lower degree space's mapping.

## 6. Binary mapping + BIHT
data set $(x,y)^d$, where $y_i=\{1, 0\}^L$

we want to map y into lower space by $$z = [M\cdot y]$$ where M is a multivariant i,i,d Gaussian matrix, and $[]$ is tkaing the sign.

Then we train binary classifiers on each bit of $z \in \{0, 1\}^{\hat L}$

For each test point, we predict its $\hat z$ and then recover the $\hat y$ using BIHT algorithm:

$$
\begin{align}
&y^0=0^L\\
&\text{for t = 0, 1, 2 ...}\\
&\quad a^{t+1}=y^t+\frac{\tau}{2}M^T(z-[My^t])\\
&\quad y^{t+1}=\eta_k(a^{t+1})
\end{align}
$$

Where $\eta_k()$ here is keeping the k greatest positive value inside a and leaves others to 0. This algorithm can converge to the recovered $\hat y$



