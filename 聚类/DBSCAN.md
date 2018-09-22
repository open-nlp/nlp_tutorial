####2.3、基于密度的聚类
基于密度的聚类的算法是数据挖掘技术中被广泛应用的一类方法，其核心思想是用一个点的 $\epsilon$ 邻域内的邻居点数量来衡量该点所在的空间密度。它可以找出形状不规则的聚类，并且聚类时不需要事先知道聚类数。DBSCAN（Density-Based Spatial Clustering of Application with Noise）是一种典型的基于密度的聚类的方法。这一小节主要对其进行详细的介绍。

#####2.3.1、基本概念
DBSCAN算法中有两个重要参数：$\textbf{Eps}$ 和 $\textbf{MinPts}$，这是一些参考文献和工具包中常用的名字，前者为定义密度时的邻域半径，后者为定义核心点时的阈值，判断一个点是否是核心点的，就看它周边的点的个数。为了方便起见，以下将$\textbf{Eps}$ 和 $\textbf{MinPts}$ 分别记为 $\epsilon$ 和 $\mathcal{M}$，考虑数据集合，$X=\{\textbf{x}^{(1)},\textbf{x}^{(2)},\textbf{x}^{(3)},...,\textbf{x}^{(N)}\}$，首先引入以下概念和记号。

**1.$\epsilon$ 邻域**
设 $\textbf{x}\in X$，称
$$\begin{equation}
N_\epsilon(\textbf{x})=\{\textbf{y}\in X:d\,(\textbf{y,x})\leq \epsilon\} \tag{2.3.1.1}
\end{equation}$$
为 $\textbf{x}$ 的 $\epsilon$ 邻域。显然，$\textbf{X} \in N_\epsilon (\textbf{x})$，有时为了简单起见，也将节点 $\textbf{x}^{(i)}$ 与其指标 $i$ 视为等同，并引入记号：
$$\begin{equation}
N_\epsilon(i)=\{j:d\,(\textbf{y}^{(j)},\textbf{x}^{(i)})\leq \epsilon ,\;\textbf{y}^{(j)},\textbf{x}^{(i)} \in X\} \tag{2.3.1.2}
\end{equation}$$
**2.密度**
设 $\textbf{x}\in X$，称
$$\begin{equation}
\rho(\textbf{x}) = \left|\;N_\epsilon(\textbf{x})\;\right| \tag{2.3.1.3}
\end{equation}$$
为 $\textbf{x}$ 的密度。注意，这里的密度是一个整数值，且依赖于半径 $\epsilon$

**3.核心点**
设 $\textbf{x}\in X$，如果$\rho(\textbf{x}) \geq \mathcal{M}$，则称 $\textbf{x}$ 为 $X$ 的核心点，由 $X$ 中所有核心点构成的集合为 $X_c$ ，并记 $X_{nc} = X \backslash X_{c}$ 表示由 $X$ 中的所有为核心点构成的集合。

**4.边界点**
如果 $\textbf{x} \in X_{nc}$ ，并且 $\exists\;\textbf{y}\in X$ ,满足：
$$\textbf{y}\in N_{\epsilon}(\textbf{x}) \cap X_c$$
则 $\textbf{x}$ 的 $\epsilon$ 邻域中存在的核心点，则称 $\textbf{x}$ 为 $X$ 的边界点。记由 $X$ 中所有边界点构成的集合为 $X_{bd}$ 。此外，边界点也可以这样定义，如果 $\textbf{x} \in X_{nc}$ ,并且 $\textbf{x}$ 落在某个核心点的 $\epsilon$ 邻域内，那么这样的点就是边界点。一个边界点可能同时落入一个或者多个核心点的 $\epsilon$ 邻域。

**5.噪音点**
现在定义噪音点，$X_{noi} = X \backslash(X_c\cup X_{bd})$ ,如果 $\textbf{x} \in X_{noi}$ ,则称 $\textbf{x}$ 为噪音点。到这里我们已经严格的给出了核心点，边界点和噪音点的数学定义，且满足 $X=X_c \cup X_{bd} \cup X_{noi}$ 。为了直观的理解，我们给定如下图示：
<center><img src="./img/2.png"/></center>.

通俗地讲，核心点对应稠密区域内部的点，边界点对应稠密区域边缘的点，而噪音点对应与系数区域中的点，如下图所示：在下图中，大点表示核心点，小点与核心点相同颜色表示边界点，黑色的小点表示噪音点。需要注意的是，核心点位于族的内部，它确定无误地属于某个特定的族，噪音点是数据集中的干扰数据，它不属于任何一个族，而边界点是一类特殊的点，它位于一个或几个族的边缘地带，可能属于一个族，也可能属于另一个族，其族归属并不明确。
<center><img src="./img/3.png"/></center>.

**6.直接密度可达**
设 $\textbf{x},\textbf{y} \in X$，若满足 $\textbf{x} \in X_c，\textbf{y} \in N_\epsilon(\textbf{x})$，则称 $\textbf{y}$ 是从 $\textbf{x}$ 直接密度可达的

**7.密度可达**
设 $\textbf{p}^{(1)},\textbf{p}^{(2)},...,\textbf{p}^{(m)} \in X$，其中 $m \geq 2$，若他们满足：$\textbf{p}^{(i+1)}$ 是从 $\textbf{p}^{(i)}$ 直接密度可达的，$i=1,2,...,m-1$，则称 $\textbf{p}^{(m)}$ 是从 $\textbf{p}^{(1)}$ 密度可达的。也就是说可以借助别人来实现密度可达。

**注**，当 $m=2$ 时，密度可达就是直接密度可达，由此可知，密度可达是直接密度可达的一种推广，事实上，密度可达是直接密度可达的 **传递闭包**

**注**，密度可达关系 **不具备对称性**，即，若 $\textbf{p}^{(m)}$ 是从 $\textbf{p}^{(1)}$ 密度可达的，则 $\textbf{p}^{(1)}$ 不一定是从 $\textbf{p}^{(m)}$ 密度可达的，因此从上述定义可知，$\textbf{p}^{(1)},\textbf{p}^{(2)},...,\textbf{p}^{(m-1)}$ 必须是核心点，而 $\textbf{p}^{(m)}$ 可以是核心点也可以是边界点。当 $\textbf{p}^{(m)}$ 为边界点时，$\textbf{p}^{(1)}$ 不可能是从 $\textbf{p}^{(m)}$ 密度可达的。**这说明，这种图是有方向的**。

**8.密度相连**
设 $\textbf{x,y,z} \in X$，若 $\textbf{y}$ 和 $\textbf{z}$ 均是从 $\textbf{x}$ 密度可达的，则称 $\textbf{y}$ 和 $\textbf{z}$ 是密度相连的。显然密度相连 **具有对称性**

**9.类**
称非空结合 $C \subset X$ 是 $X$ 的一个类，如果满足：对 $\textbf{x,y} \in X$
+ $(Maximality)$ 若 $\textbf{x} \in C$，且 $\textbf{y}$ 是从 $\textbf{x}$ 密度可达的，则 $\textbf{y} \in C$

+ $(Connectivity)$ 若 $\textbf{x} \in C$ , $\textbf{y} \in C$ ,则称$\textbf{x，y}$ 是密度相连的。

#####2.3.2、算法描述
这一部分主要介绍DBSCAN算法，其核心思想可描述如下：**从某个选定的核心点出发，不断向密度可达的区域扩张，从而得到一个包含核心点和边界点的最大化区域，区域中任意两点密度相连**

考虑数据集合 $X=\{\textbf{x}^{(1)},\textbf{x}^{(2)},...,\textbf{x}^{(N)}\}$，DBSCAN算法的目标是将数据集合 $X$ 分成 $K$ 个 $cluster$ **(注意 $K$ 也是由算法得到，无需事先指定)**以及噪音点集合，引入 $cluster$ 标记数组：
$$\begin{equation}
m_i=\left\{
   \begin{aligned}
   j\quad(j\gt 0) \quad,若 \textbf{x}^{(i)} 属于第\;j\;个\;cluster\\
   -1 \qquad\qquad ,若 \textbf{x}^{(i)} 为噪音点\qquad\quad\;\;\\
   \end{aligned}
   \right.
\end{equation}$$
由此，DBSCAN算法的目标就是生成标记数组，$m_i,i=1,2,...,N$，而 $K$ 即为 $\{m_i\}^N_{i=1}$ 中互异的非负的个数，下面就给出了DBSCAN算法的描述

**算法：DBSCAN**
+ **Step 1** 初始化

  1. 给定参数 $\epsilon$ 和 $\mathcal{M}$

  2. 生成 $N_\epsilon(i)，i=1,2,...,N$，生成所有数据的邻域点

  3. 令 $k=1;m_i=0,i=1,2,...,N$，首先见所有数据的标签设置为0

  4. 令 $I=\{1,2,...,N\}$
+ **Step 2** 生成 $cluster$ 标记数组

  1. 如果，$I$ 不为空，从 $I$ 中任取一个元素 $i$，并令 $I=I\backslash\{i\}$
  
  2. 如果 $m_i=0$，也就是节点 $i$ 还没有被处理过，则进行下面3-5步，否则重复1
  
  3. 初始化 $T=N_\epsilon(i)$
  
  4. 若 $|T| \lt \mathcal{M}$，则令 $m_i=-1$，（暂时将 $i$ 号节点标记为 **噪音点**）
  
  5. 若 $|T| \ge \mathcal{M}$，则 $i$ 为核心点，则进行以下三个步骤：
  
    a. 令 $m_i = k$，将 $i$ 号节点归属于第 $k$ 个聚类
    
    b. 如果 $T \neq \emptyset$，则进行以下三个步骤
    
      （1）从 $T$ 中任取元素 $j$，并令 $T=T\backslash \{j\}$
      
      （2）若 $m_j=0$ 或 -1，则令 $m_j=k$
      
      （3）若 $|N_\epsilon(j)|\geq \mathcal{M}$，也就是 $j$ 为核心点，则令 $T=T\cup N_\epsilon(j)$
      
    c. 令 $k=k+1$，表示第 $k$ 个聚类以及完成，开始下一个聚类。

**注** **瓶颈问题**
上面算法的瓶颈在于 $N_\epsilon(i),\;i=1,2,...,N$ 的计算，一种做法是引入一个（对称的）距离矩阵，用来存储 $X$ 中，任意两个节点之间的距离，但这种方法的时间复杂度为 $O(N^2)$，并且需要 $O(\frac{1}{2}N^2)$ 的存储开销，对大规模数据集是不可行的。

为了提升效率，可以建立指标索引，如 $R-tree$ 或 $k-d tree$ 。此外，也可以基于网格来做，该方法用网格划分的方法和数据分箱技术减少判定密度可达对象时的搜索范围。

注意，算法描述中为方便起见，将 $N_\epsilon(i)$ 的计算放在初始步，这意味着实现开设足够的空间存放他们，然而随着 $\epsilon$ 的增大，这种存储开销趋于 $O(N^2)$，因此，在实际变成中，$N_\epsilon(i)$ 的查询嵌套在step 2中，从而节省存储空间。

**注** **噪音点的确定**
在 step 2 中，将满足 $|T|\lt \mathcal{M}$ 的节点 $\textbf{x}^{(i)}$ 的标记属性赋值为 $-1$ ，事实上这里面处理真正的噪音点，还可能包括一些边界点，但是在上面算法中的 $5-b-(2)$ 中，又将这些边界点归到相应的 $cluster$ 中。

**注** **由一个核心点确定一个 $cluster$ **
在 step 2 中，对应的是由一个核心点（也成为“种子点”）确定一个 $cluster$ 的过程，注意 $T$ 是用来存放待加入 $\textbf{x}^{(i)}$ 所在 $cluster$ 的候选集合的容器，在循环中，它是动态变化的，因此可以通过栈结构来实现。此外，采用递归的方法也是可行的。

**注** **边界点的归属问题**
前面我们提到，边界点是一类特殊的点，它位于一个或几个 $cluster$ 的边缘地带，因此其 $cluster$ 归属并不明确，但在算法中，我们可以看到，所有边界点都唯一地归属到了一个 $cluster$，显然，这完全取决于节点遍历的顺序。

当边界点的归属很重要时，我们可以在 step 2 中，先将边界点标记出来（不划分任何 cluster），然后增加 step 3 来单独处理他们，如何处理呢？这里提供一个策略：（1）计算各 $cluster$ 的中心点 $\textbf{y}^{(j)}$; (2)按照公式 $m_i = arg\; min \;d(\textbf{x}^{(i)},\textbf{y}^{(j)})$ 对边界点 $\textbf{x}^{(i)}$ 进行归来，事实上，这里用的是 $K-means$ 算法的思想。

接下来，我们进一步讨论DBSCAN算法的几个重要问题：

1. **参数 $\epsilon$ 选择**
DBSCAN算法中使用了统一的 $\epsilon$ 值，因此数据空间中国你所有节点的领域大小是一致的，当数据密度和 $cluster$ 间距离分布不均匀时，若选取较小的 $\epsilon$ 值，则较稀疏 $cluster$ 中的节点的密度会小于 $\mathcal{M}$ ，从而被认为是边界点而不被用于所在类的进一步扩展，随之而来的结果是，较稀的 $cluster$ 可能被划分成多个性质相似的 $cluster$ ，与此相反，若选取较大的 $\epsilon$ 值，则离得比较近而密度比较大的那些 $cluster$ 将很可能被合并为同一个 $cluster$ ，他们之间的差异将被忽略，显然，在这种情况下，要选取一个合适的 $\epsilon$ 值比较困难，对于高维数据，由于“维数灾难”，$\epsilon$ 的合理选择变得更加困难。

2. **参数 $\mathcal{M}$ 选择**
$\mathcal{M}$ 的选择有一个指导性的原则，也就是 $\mathcal{M} \geq dim +1$，这里 $dim$ 表示聚类数据空间的维度（也就是 $X$ 中元素的长度）

3. **复杂度问题**
DBSCAN算法需要访问 $X$ 中所有节点（有些节点，如边界节点，可能还需访问多次），因此事件复杂度主要取决于 **区块查询** （获取某个节点的 $\epsilon$ 邻域）的次数，DBSCAN算法的事件复杂度为 $O(N^2)$，如果使用 $k-d$ 树等索引结果，复杂度可将为 $O(Nlog N)$，注意，在复杂度为 $O(N^2)$ 的算法中，为了避免重复计算距离，通常事先生成一个 $N$ 阶的对称距离矩阵，这样存储该矩阵也需要 $O(N^2)$ 的空间开销。


#####2.3.3、优缺点
+ **优点**
  1. 不需要事先指定cluster的数量

  2. 可以发现任何形状的cluster，聚类效果如图：
<center><img src="./img/4.png"/></center>.

  3. 能找出数据中的噪音，且对噪音不敏感

  4. 算法中只需要确定两个参数

  5. 聚类结果几乎不依赖于节点的遍历顺序

+ **缺点**

  1. DBSCAN 算法的聚类质量（或效果）依赖于距离公式的选取，实际应用中，最常见的距离公式是欧几里得距离，然而，对于高纬度数据，由于“维度灾难” 的影响，距离的度量标准变得不再重要。

  2. DBSCAN算法不适合与数据集密度差异很大的情况，在这种情况下，参数 $\epsilon$ 和 $\mathcal{M}$ 的选取是很困难的。

