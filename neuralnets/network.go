package main

type NumericInput struct {
	Number float64
}

func (n *NumericInput) Value() float64 {
	return n.Number
}

type Network struct {
	inputs  []*NumericInput
	outputs []*Neuron
}

func NewNetwork(in, out, hiddenLayers, branchFactor int, f ActivationFunction) *Network {
	var res Network

	res.inputs = make([]*NumericInput, in)
	inputsInterface := make([]Input, in)
	for i := 0; i < in; i++ {
		res.inputs[i] = &NumericInput{}
		inputsInterface[i] = res.inputs[i]
	}

	for i := 0; i < out; i++ {
		output := generateOutput(inputsInterface, hiddenLayers+1, branchFactor, f)
		res.outputs = append(res.outputs, output)
	}
	return &res
}

func (n *Network) SetInput(values []float64) {
	for i, x := range values {
		n.inputs[i].Number = x
	}
}

func (n *Network) Adjust(desired []float64, velocity float64) {
	for i, output := range n.outputs {
		output.BackPropagate(desired[i], velocity)
	}
}

func (n *Network) Evaluate() []float64 {
	res := make([]float64, len(n.outputs))
	for i, output := range n.outputs {
		res[i] = output.Value()
	}
	return res
}

func generateOutput(inputs []Input, layers, branchFactor int, f ActivationFunction) *Neuron {
	if layers == 0 {
		return NewNeuron(inputs, f)
	}
	upstreamInputs := make([]Input, branchFactor)
	for i := 0; i < branchFactor; i++ {
		upstreamInputs[i] = generateOutput(inputs, layers-1, branchFactor, f)
	}
	return NewNeuron(upstreamInputs, f)
}
