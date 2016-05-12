package convnet

import (
	"math"
	"math/rand"
)

// SGD trains neural networks using
// gradient descent.
type SGD struct {
	CostFunc CostFunc
	Inputs   [][]float64
	Outputs  [][]float64

	// StepSize indicates how far along each
	// gradient the solver should move.
	StepSize float64

	// StepDecreaseRate specifies how much to
	// decrease the step size after every epoch
	// of gradient descent.
	StepDecreaseRate float64

	// Epochs is the number of rounds of descent
	// the solver should perform before stopping.
	Epochs int
}

func (s *SGD) Train(n *Network) {
	downstreamGrad := make([]float64, len(n.Output()))
	n.SetDownstreamGradient(downstreamGrad)
	for i := 0; i < s.Epochs; i++ {
		stepSize := s.StepSize - float64(i)*s.StepDecreaseRate
		if stepSize <= 0 {
			break
		}

		order := rand.Perm(len(s.Inputs))
		for _, j := range order {
			input := s.Inputs[j]
			output := s.Outputs[j]
			n.SetInput(input)
			n.PropagateForward()
			s.CostFunc.Deriv(downstreamGrad, n.Output(), output)
			propagateBackwardsAlmost(n)

			grad := math.Sqrt(n.GradientMagSquared())
			n.StepGradient(stepSize / grad)
		}
	}
}

func propagateBackwardsAlmost(n *Network) {
	for i := len(n.Layers) - 1; i > 0; i-- {
		n.Layers[i].PropagateBackward()
	}
}
