package main

import "math/rand"

type Input interface {
	Value() float64
}

type Neuron struct {
	weights    []float64
	inputs     []Input
	activation ActivationFunction

	sumCache   float64
	valueCache float64
}

func NewNeuron(inputs []Input, activation ActivationFunction) *Neuron {
	randomWeights := make([]float64, len(inputs)+1)
	for i := range randomWeights {
		randomWeights[i] = rand.Float64()*10.0 - 5.0
	}

	return &Neuron{
		weights:    randomWeights,
		inputs:     inputs,
		activation: activation,
	}
}

func (n *Neuron) Value() float64 {
	n.sumCache = 0
	for i, weight := range n.weights {
		n.sumCache += weight * n.inputValue(i)
	}
	n.valueCache = n.activation.Evaluate(n.sumCache)
	return n.valueCache
}

func (n *Neuron) BackPropagate(desired, velocity float64) {
	actual := n.Value()
	errorPerOutput := (desired - actual)
	n.backPropagate(errorPerOutput, velocity)
}

func (n *Neuron) backPropagate(errorPerOutput, velocity float64) {
	errorPerSum := errorPerOutput * n.activation.EvaluateDerivative(n.sumCache)
	oldWeights := make([]float64, len(n.weights))
	copy(oldWeights, n.weights)
	for i, weight := range n.weights {
		errorPerWeight := n.inputValue(i) * errorPerSum
		n.weights[i] = weight + errorPerWeight*velocity
	}
	for i, input := range n.inputs {
		if neuron, ok := input.(*Neuron); ok {
			newErrorPerOutput := errorPerSum * oldWeights[i+1]
			neuron.backPropagate(newErrorPerOutput, velocity)
		}
	}
}

func (n *Neuron) inputValue(i int) float64 {
	// NOTE: the first input value is the threshold.
	if i == 0 {
		return -1
	} else {
		return n.inputs[i-1].Value()
	}
}
