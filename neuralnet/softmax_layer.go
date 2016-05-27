package neuralnet

import (
	"math"
	"strconv"

	"github.com/unixpickle/num-analysis/kahan"
)

type SoftmaxParams struct {
	Size int
}

// Make creates a new *SoftmaxLayer using
// the parameters specified in s.
// This is equivalent to NewSoftmaxLayer(s).
func (s *SoftmaxParams) Make() Layer {
	return NewSoftmaxLayer(s)
}

type SoftmaxLayer struct {
	output           []float64
	exponentials     []float64
	upstreamGradient []float64

	input              []float64
	downstreamGradient []float64

	lastExpSum float64
}

func NewSoftmaxLayer(p *SoftmaxParams) *SoftmaxLayer {
	return &SoftmaxLayer{
		output:           make([]float64, p.Size),
		upstreamGradient: make([]float64, p.Size),
		exponentials:     make([]float64, p.Size),
	}
}

func DeserializeSoftmaxLayer(d []byte) (*SoftmaxLayer, error) {
	size, err := strconv.Atoi(string(d))
	if err != nil {
		return nil, err
	}
	return NewSoftmaxLayer(&SoftmaxParams{Size: size}), nil
}

func (s *SoftmaxLayer) Output() []float64 {
	return s.output
}

func (s *SoftmaxLayer) UpstreamGradient() []float64 {
	return s.upstreamGradient
}

func (s *SoftmaxLayer) Input() []float64 {
	return s.input
}

func (s *SoftmaxLayer) SetInput(v []float64) bool {
	if len(v) != len(s.upstreamGradient) {
		return false
	}
	s.input = v
	return true
}

func (s *SoftmaxLayer) DownstreamGradient() []float64 {
	return s.downstreamGradient
}

func (s *SoftmaxLayer) SetDownstreamGradient(v []float64) bool {
	if len(v) != len(s.output) {
		return false
	}
	s.downstreamGradient = v
	return true
}

func (s *SoftmaxLayer) Randomize() {
}

func (s *SoftmaxLayer) PropagateForward() {
	var totalSum kahan.Summer64
	for i, x := range s.input {
		exp := math.Exp(x)
		s.exponentials[i] = exp
		totalSum.Add(exp)
	}
	normalizer := 1.0 / totalSum.Sum()
	for i, x := range s.exponentials {
		s.output[i] = x * normalizer
	}
	s.lastExpSum = totalSum.Sum()
}

func (s *SoftmaxLayer) PropagateBackward(upstream bool) {
	if !upstream {
		return
	}

	sum := s.lastExpSum
	normalizer := 1 / sum
	for i, x := range s.exponentials {
		var partial kahan.Summer64

		outProb := s.output[i]

		restSum := sum - x
		selfPartial := outProb * normalizer * restSum
		partial.Add(selfPartial * s.downstreamGradient[i])

		for j, x := range s.exponentials {
			if j == i {
				continue
			}
			otherPartial := -outProb * normalizer * x
			partial.Add(otherPartial * s.downstreamGradient[j])
		}

		s.upstreamGradient[i] = partial.Sum()
	}
}

func (s *SoftmaxLayer) GradientMagSquared() float64 {
	return 0
}

func (s *SoftmaxLayer) StepGradient(f float64) {
}

func (s *SoftmaxLayer) Alias() Layer {
	return NewSoftmaxLayer(&SoftmaxParams{Size: len(s.output)})
}

func (s *SoftmaxLayer) Serialize() []byte {
	return []byte(strconv.Itoa(len(s.output)))
}

func (s *SoftmaxLayer) SerializerType() string {
	return "softmaxlayer"
}
