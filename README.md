# weakai

I have been learning about various weak AI algorithms. In this repository, I will implement what I have learned and document the sources of my knowledge.

# What's included

Here are the Go packages I've created for AI:

 * [svm](svm) - an implementation of Support Vector Machines, complete with my own solver. I am no expert at numerical analysis or quadratic optimization, but my solver works fairly well.
 * [neuralnet](neuralnet) - a Go library for feed-forward neural networks.
 * [rbm](rbm) - Restricted Boltzmann Machine sampler and trainer.
 * [idtrees](idtrees) - a general identification tree implementation with an accompanying command-line tool that parses CSV files. This includes sample data about various celebrities.
 * [boosting](boosting) - an implementation of two different boosting algorithms: AdaBoost and Gradient Boosting.
 * [rnn](rnn) - a Recurrent Neural Networks library.

Here are some demo programs I've created:

 * [objectrecog](demos/objectrecog) - an implementation of correlative object recognition. First, you show it an object from your webcam, then it finds that object in other pictures. This works surprisingly well for face tracking. This is a web application intended for desktops, since it does not support touch screens and most mobile devices do not support the `getUserMedia()` API.
 * [hopfield](demos/hopfield) - a graphical (HTML) demonstration of [Hopfield networks](https://en.wikipedia.org/wiki/Hopfield_network).
 * [mapcolor](demos/mapcolor) - four-color a map of the USA using a constraint search. This is a Go program that modifies an SVG of the USA and outputs the result.
 * [nearestneighbors](demos/nearestneighbors) - a simple search engine that uses Nearest Neighbors. The search engine itself is far from useful, but at least it demonstrates a technique of Nearest Neighbors learning.
 * [minimax](demos/minimax) - checkers AI that uses the minimax algorithm. This is an HTML+CSS+SVG+JavaScript application.

# Sources

 * SVMs, ANNs, idtrees, boosting, minimax, constraints, object recognition, and kNN
   * [MIT's OpenCourseware AI lectures](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/)
   * [deeplearning.net on ConvNets](http://deeplearning.net/tutorial/lenet.html#lenet)
 * Restricted Boltzmann Machines
   * [RBM lecture](https://www.youtube.com/watch?v=FJ0z3Ubagt4)
   * [Geoffrey Hinton lecture](https://www.youtube.com/watch?v=tt-PQNstYp4)
 * Recurrent Neural Networks
   * [RNN Lecture (Part 1)](https://www.youtube.com/watch?v=AvyhbrQptHk)
   * [RNN Lecture (Part 2)](https://www.youtube.com/watch?v=EAt9_4IhC7s)
   * [Wikipedia on LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory)
   * [GRU Paper](http://arxiv.org/pdf/1406.1078v3.pdf)
   * [Karpathy's char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
