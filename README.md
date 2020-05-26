# Decision Tree
This assignment aims to familiarize you with the mechanism of a widely-used classification methods: decision tree (DT)

Specifically, the problem in this assignment is a multi-class classification problem with continuous feature/attribute values. For the decision tree you are to implement, please always use binary split and a threshold to split data. That is, each decision node has the form:

    If attribute X ≤ threshold θ; then left node; else right node.
where the best X and θ are for you to determine. Please use Gini impurity to construct the
decision tree.

## Model specifications
- For DT, let max depth = 2 be the only stopping criterion. 

In other words, you should grow the tree as long as max depth = 2 is not violated and not all training instances in the current node have the same label. In face of multiple choices of threshold θ with the same resulting Gini impurity, always choose the one with the largest margin. (We will illustrate this point with example at a later stage.)

**The following design choice in implementation purely aims to ease auto- grading on HackerRank.** 

We ensure each attribute is named by a **non-negative integer**, and each class label is named by a **positive integer**. Since we are to use HackerRank for grading, we have to eliminate additional randomness and generate the deterministic results. We therefore enforce the following rule in this assignment: In the event of ties, always choose the attribute or label with the smallest value. Namely,
- When training a DT, if splitting on either attribute X1 or X2 gives you the best Gini impurity, choose the smaller of X1 and X2.
- In prediction, if both label L1 and L2 have the same number of training instances at a leaf node of a DT, predict the smaller of L1 and L2.

## Input Format and Sample
Each input dataset contains training instances followed by test instances in a format adapted from the libsvm format. Each line has the form

    [label] [attribute 1]:[value 1] [attribute 2]:[value 2] . . .
which is space-separated. As mentioned above, the name of each attribute, e.g., [attribute 2], is a non-negative integer and the value of an attribute, e.g., [value 2], is a float number. A line stands for a **test instance if [label] is 0** and a training instance otherwise. The label of a training instance can be any positive integer number.

Again, you may assume the test instances ([label] = 0) are at the end of the input file. Note that please do not assume the attribute names to start from 0 or to be consecutive integers, and please do not assume the class labels to start from 1 or to be consecutive integers.

    1 0:1.0 2:1.0 
    1 0:1.0 2:2.0 
    1 0:2.0 2:1.0 
    3 0:2.0 2:2.0 
    1 0:3.0 2:1.0 
    3 0:3.0 2:2.0 
    3 0:3.0 2:3.0 
    3 0:4.5 2:3.0 
    0 0:1.0 2:2.2 
    0 0:4.5 2:1.0

## Output Format and Sample
The output is the prediction on the test instances made by your DT. Print the prediction for each test instance per line, following the order in the input file.
As an example the output of the toy example is as follows.

    1
    1
