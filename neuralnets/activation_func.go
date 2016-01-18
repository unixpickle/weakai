package main

import "math"

type ActivationFunction interface {
	Evaluate(x float64) float64
	EvaluateDerivative(x float64) float64
}

type SigmoidFunction struct{}

func (s SigmoidFunction) Evaluate(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func (s SigmoidFunction) EvaluateDerivative(x float64) float64 {
	return s.Evaluate(x) * (1 - s.Evaluate(x))
}
