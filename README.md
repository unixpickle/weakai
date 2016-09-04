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

# Projects which use this

Here are the projects I have created which use something in the `weakai` repository.

 * [char-rnn](https://github.com/unixpickle/char-rnn) - train RNNs to create text
 * [seqtasks](https://github.com/unixpickle/seqtasks) - benchmarks for RNNs
 * [neuralstruct](https://github.com/unixpickle/neuralstruct) - attach data structures to RNNs
 * [whichlang](https://github.com/unixpickle/whichlang) - classify programming languages
 * [samepic](https://github.com/unixpickle/samepic) - tell if images are of the same thing
 * [neuraltree](https://github.com/unixpickle/neuraltree) - tree-based neural architecture
 * [spacesplice](https://github.com/unixpickle/spacesplice) - add spaces to text without spaces
 * [haar](https://github.com/unixpickle/haar) - visual object detection
 * [speechrecog](https://github.com/unixpickle/speechrecog) - general RNN-based speech recognition
   * [cubewhisper](https://github.com/unixpickle/cubewhisper) - speech recognition for Rubik's cube moves
 * [mnistdemo](https://github.com/unixpickle/mnistdemo) - MNIST classifiers in action
 * [svm-playground](https://github.com/unixpickle/svm-playground) - SVMs in action
 * [statebrain](https://github.com/unixpickle/statebrain) - trainable Markov models
 * [humancube](https://github.com/unixpickle/humancube) - generate human-like Rubik's cube moves
 * [sentigraph](https://github.com/unixpickle/sentigraph) - graph sentiment over a piece of text

# Sources

 * SVMs, ANNs, idtrees, boosting, minimax, constraints, object recognition, and kNN
   * [MIT's OpenCourseware AI lectures](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/)
 * Convolutional networks
   * [deeplearning.net on ConvNets](http://deeplearning.net/tutorial/lenet.html#lenet)
   * [AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
   * [CNN Paper from 2015](https://arxiv.org/pdf/1409.1556.pdf)
   * [CNN Paper from 2013](http://arxiv.org/pdf/1311.2901v3.pdf)
 * Restricted Boltzmann Machines
   * [RBM lecture](https://www.youtube.com/watch?v=FJ0z3Ubagt4)
   * [Geoffrey Hinton lecture](https://www.youtube.com/watch?v=tt-PQNstYp4)
 * Recurrent Neural Networks
   * [RNN Lecture (Part 1)](https://www.youtube.com/watch?v=AvyhbrQptHk)
   * [RNN Lecture (Part 2)](https://www.youtube.com/watch?v=EAt9_4IhC7s)
   * [Wikipedia on LSTMs](https://en.wikipedia.org/wiki/Long_short-term_memory)
   * [GRU Paper](http://arxiv.org/pdf/1406.1078v3.pdf)
   * [Karpathy's char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
   * [Bidirectional RNNs](http://arxiv.org/pdf/1303.5778.pdf)
   * [Identity RNNs](https://arxiv.org/pdf/1504.00941v2.pdf)
   * [npRNNs](http://arxiv.org/pdf/1511.03771v3.pdf)
