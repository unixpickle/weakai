# weakai

I have begun watching some of [MIT's OpenCourseware AI lectures](http://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-034-artificial-intelligence-fall-2010/). In this repository, I will attempt to implement what I've learned.

# What's included

Here are the weak AI demos I've implemented:

 * [minimax](minimax) - checkers AI that uses the minimax algorithm. This is an HTML+CSS+SVG+JavaScript application.
 * [neuralnets](neuralnets) - a Go implementation of basic neural networks and back-propagation. I wrote two tests for the neural net, the second of which demonstrates ANNs' ability to generalize what they've learned.
 * [mapcolor](mapcolor) - four-color a map of the USA using a constraint search. This is a Go program that modifies an SVG of the USA and outputs the result.
 * [objectrecog](objectrecog) - an implementation of correlative object recognition. First, you show it an object from your webcam, then it finds that object in other pictures. This works surprisingly well for face tracking. This is a web application intended for desktops, since it does not support touch screens and most mobile devices do not support the `getUserMedia()` API.
 * [nearestneighbors](nearestneighbors) - a simple search engine that uses Nearest Neighbors. The search engine itself is far from useful, but at least it demonstrates a technique of Nearest Neighbors learning.
 * [idtrees](idtrees) - a general identification tree implementation with an accompanying command-line tool that parses CSV files. This includes sample data about various celebrities.
