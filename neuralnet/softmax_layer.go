package main

import (
	"math"
	"strconv"

	"github.com/unixpickle/num-analysis/kahan"
)

type SoftmaxParams struct {
	Size int
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
	}
}

func DeserializeSoftmaxLayer(d []byte) (*SoftmaxLayer, error) {
	size, err := strconv.Atoi(string(d))
	if err != nil {
		return nil, err
	}
	return NewSoftmaxLayer(&SoftmaxParams{Size: size})
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
}

func (s *SoftmaxLayer) DownstreamGradient() []float64 {
	return s.downstreamGradient
}

func (s *SoftmaxLayer) SetDownstreamGradient(v []float64) bool {
	if len(v) != len(s.output) {
		return false
	}
	s.downstreamGradient = v
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
	sum := s.lastExpSum
	normalizer := 1 / sum
	for i, x := range s.exponentials {
		var partial kahan.Summer64

		restSum := sum - x
		selfPartial := s.output[i] * normalizer * restSum
		partial.Add(selfPartial * s.downstreamGradient[i])

		for j, x := range s.output {
			if j == i {
				continue
			}
			otherPartial := -math.Pow(x, 2) * s.exponentials[j]
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

func (s *SoftmaxLayer) Serialize() []byte {
	return []byte(strconv.Itoa(len(s.output)))
}

func (s *SoftmaxLayer) SerializerType() string {
	return "softmaxlayer"
}
