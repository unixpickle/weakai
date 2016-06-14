package rnntest

import (
	"math"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

const (
	gradientTestDelta = 1e-6
	gradientTestPrec  = 1e-6
)

// GradientTest performs gradient and r-gradient
// checking on a Block.
type GradientTest struct {
	Block          rnn.Block
	GradientParams []*autofunc.Variable
	Inputs         []*autofunc.Variable
	InStates       []*autofunc.Variable
}

func (g *GradientTest) Run(t *testing.T) {
	actual := exactJacobian(g.evaluate, g.GradientParams)
	expected := approximateJacobian(g.evaluate, g.GradientParams)

	if len(actual) != len(expected) {
		t.Fatal("jacobian lengths don't match")
	}

	for i, x := range expected {
		a := actual[i]
		if len(x) != len(a) {
			t.Fatal("gradient lengths don't match")
		}
		for variableIdx, variable := range g.GradientParams {
			expVec := x[variable]
			actVec := a[variable]
			if len(expVec) != len(actVec) {
				t.Fatal("vector lengths don't match")
			}
			for j, expVal := range expVec {
				actVal := actVec[j]
				if math.Abs(expVal-actVal) > gradientTestPrec {
					t.Errorf("output %d variable %d: expected %f got %f (idx %d)",
						i, variableIdx, expVal, actVal, j)
				}
			}
		}
	}
}

func (g *GradientTest) evaluate() rnn.BlockOutput {
	return g.Block.Batch(&rnn.BlockInput{Inputs: g.Inputs, States: g.InStates})
}

func dividedDifference(val *float64, f func() float64) float64 {
	old := *val
	*val = old + gradientTestDelta
	rightVal := f()
	*val = old - gradientTestDelta
	leftVal := f()
	*val = old

	return (rightVal - leftVal) / (2 * gradientTestDelta)
}

func approximateJacobian(f func() rnn.BlockOutput, v []*autofunc.Variable) []autofunc.Gradient {
	var res []autofunc.Gradient
	tmp := f()
	for i, out := range tmp.Outputs() {
		for j := range out {
			grad := autofunc.Gradient{}
			for _, variable := range v {
				varGrad := make(linalg.Vector, len(variable.Vector))
				for k := range variable.Vector {
					varGrad[k] = dividedDifference(&variable.Vector[k], func() float64 {
						x := f()
						return x.Outputs()[i][j]
					})
				}
				grad[variable] = varGrad
			}
			res = append(res, grad)
		}
	}
	for i, out := range tmp.States() {
		for j := range out {
			grad := autofunc.Gradient{}
			for _, variable := range v {
				varGrad := make(linalg.Vector, len(variable.Vector))
				for k := range variable.Vector {
					varGrad[k] = dividedDifference(&variable.Vector[k], func() float64 {
						x := f()
						return x.States()[i][j]
					})
				}
				grad[variable] = varGrad
			}
			res = append(res, grad)
		}
	}
	return res
}

func exactJacobian(f func() rnn.BlockOutput, v []*autofunc.Variable) []autofunc.Gradient {
	var res []autofunc.Gradient
	tmp := f()
	for i, out := range tmp.Outputs() {
		for j := range out {
			grad := autofunc.Gradient{}
			for _, variable := range v {
				grad[variable] = make(linalg.Vector, len(variable.Vector))
			}
			upstream := &rnn.UpstreamGradient{
				Outputs: make([]linalg.Vector, len(tmp.Outputs())),
			}
			upstream.Outputs[i][j] = 1
			tmp.Gradient(upstream, grad)
			res = append(res, grad)
		}
	}
	for i, out := range tmp.States() {
		for j := range out {
			grad := autofunc.Gradient{}
			for _, variable := range v {
				grad[variable] = make(linalg.Vector, len(variable.Vector))
			}
			upstream := &rnn.UpstreamGradient{
				States: make([]linalg.Vector, len(tmp.States())),
			}
			upstream.States[i][j] = 1
			tmp.Gradient(upstream, grad)
			res = append(res, grad)
		}
	}
	return res
}
