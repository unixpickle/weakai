# weakai

I have been learning about various weak AI algorithms. So far, I have learned most of this stuff from [MIT's OpenCourseware AI lectures](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/), but I will soon begin looking for new sources as well. In this repository, I will attempt to implement what I've learned.

# What's included

Here are the weak AI demos I've implemented:

 * [svm](svm) - an implementation of Support Vector Machines, complete with my own solver. I am no expert at numerical analysis or quadratic optimization, but my solver works fairly well.
 * [neuralnet](neuralnet) - a Go library for feed-forward neural networks.
 * [objectrecog](objectrecog) - an implementation of correlative object recognition. First, you show it an object from your webcam, then it finds that object in other pictures. This works surprisingly well for face tracking. This is a web application intended for desktops, since it does not support touch screens and most mobile devices do not support the `getUserMedia()` API.
 * [idtrees](idtrees) - a general identification tree implementation with an accompanying command-line tool that parses CSV files. This includes sample data about various celebrities.
 * [hopfield](hopfield) - a graphical (HTML) demonstration of [Hopfield networks](https://en.wikipedia.org/wiki/Hopfield_network).
 * [boosting](boosting) - an implementation of two different boosting algorithms: AdaBoost and Gradient Boosting.
 * [minimax](minimax) - checkers AI that uses the minimax algorithm. This is an HTML+CSS+SVG+JavaScript application.
 * [mapcolor](mapcolor) - four-color a map of the USA using a constraint search. This is a Go program that modifies an SVG of the USA and outputs the result.
 * [nearestneighbors](nearestneighbors) - a simple search engine that uses Nearest Neighbors. The search engine itself is far from useful, but at least it demonstrates a technique of Nearest Neighbors learning.

# TODO

 * For idtrees implementation, add a way to prefer the floating point split that is as neutral as possible, instead of picking an arbitrary split which happens to maximize entropy. This way, if multiple splits have the same effect, the middle one is chosen.
