package rnntest

import (
	"math"
	"math/rand"
	"strconv"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

const (
	gradientTestDelta = 1e-4
	gradientTestPrec  = 1e-4
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
	g.checkGradient(t)
	g.checkRGradient(t)
	g.checkROutput(t)
}

func (g *GradientTest) checkGradient(t *testing.T) {
	for i := 0; i < 2; i++ {
		actual := exactJacobian(g.evaluate, g.GradientParams)
		var expected []autofunc.Gradient
		if i == 0 {
			expected = approximateJacobian(g.evaluate, g.GradientParams)
		} else {
			ev := g.rEvaluator(g.randomRVector())
			expected = approximateJacobian(func() rnn.BlockOutput {
				return &rOutputToOutput{ev()}
			}, g.GradientParams)
		}

		if len(actual) != len(expected) {
			t.Fatal("jacobian lengths don't match")
		}

		var actualMaps []map[*autofunc.Variable]linalg.Vector
		var expectedMaps []map[*autofunc.Variable]linalg.Vector
		for i, a := range actual {
			actualMaps = append(actualMaps, a)
			expectedMaps = append(expectedMaps, expected[i])
		}

		g.compareGradMaps(t, "checkGradient"+strconv.Itoa(i+1)+": ",
			actualMaps, expectedMaps)
	}
}

func (g *GradientTest) checkRGradient(t *testing.T) {
	rv := g.randomRVector()
	expected := approximateRJacobian(g.rEvaluator(rv), g.GradientParams, rv)
	actual := exactRJacobian(g.rEvaluator(rv), g.GradientParams)

	var actualMaps []map[*autofunc.Variable]linalg.Vector
	var expectedMaps []map[*autofunc.Variable]linalg.Vector
	for i, a := range actual {
		actualMaps = append(actualMaps, a)
		expectedMaps = append(expectedMaps, expected[i])
	}

	g.compareGradMaps(t, "checkRGradient: ", actualMaps, expectedMaps)
}

func (g *GradientTest) checkROutput(t *testing.T) {
	rv := g.randomRVector()
	routputs, rstates := approximateROutput(g.evaluate, rv)
	actualRes := g.rEvaluator(rv)()
	actualList := [][]linalg.Vector{actualRes.ROutputs(), actualRes.RStates()}
	expectedList := [][]linalg.Vector{routputs, rstates}
	for i, expected := range expectedList {
		actual := actualList[i]
		if len(expected) != len(actual) {
			t.Error("list", i, "len mismatch")
			continue
		}
		for j, x := range expected {
			a := actual[j]
			if len(x) != len(a) {
				t.Error("list", i, ",", j, "len mismatch")
				continue
			}
			for k, value := range x {
				actualVal := a[k]
				prec := gradientTestPrec
				if math.Abs(value) > 1 {
					prec = math.Abs(value) * gradientTestPrec
				}
				if math.Abs(value-actualVal) > prec {
					t.Errorf("checkROutput: idx %d,%d,%d: expected %f got %f",
						i, j, k, value, actualVal)
				}
			}
		}
	}
}

func (g *GradientTest) evaluate() rnn.BlockOutput {
	return g.Block.Batch(&rnn.BlockInput{Inputs: g.Inputs, States: g.InStates})
}

func (g *GradientTest) rEvaluator(rv autofunc.RVector) func() rnn.BlockROutput {
	return func() rnn.BlockROutput {
		var rinputs []*autofunc.RVariable
		var rstates []*autofunc.RVariable
		for _, input := range g.Inputs {
			rinputs = append(rinputs, autofunc.NewRVariable(input, rv))
		}
		for _, state := range g.InStates {
			rstates = append(rstates, autofunc.NewRVariable(state, rv))
		}
		return g.Block.BatchR(rv, &rnn.BlockRInput{Inputs: rinputs, States: rstates})
	}
}

func (g *GradientTest) randomRVector() autofunc.RVector {
	res := autofunc.RVector{}
	for _, variable := range g.GradientParams {
		vec := make(linalg.Vector, len(variable.Vector))
		for i := range vec {
			vec[i] = rand.NormFloat64()
		}
		res[variable] = vec
	}
	return res
}

func (g *GradientTest) compareGradMaps(t *testing.T, prefix string, actual,
	expected []map[*autofunc.Variable]linalg.Vector) {
	for i, x := range expected {
		a := actual[i]
		if len(x) != len(a) {
			t.Error(prefix + "gradient lengths don't match")
			return
		}
		for variableIdx, variable := range g.GradientParams {
			expVec := x[variable]
			actVec := a[variable]
			if len(expVec) != len(actVec) {
				t.Error(prefix + "vector lengths don't match")
				return
			}
			for j, expVal := range expVec {
				actVal := actVec[j]
				prec := gradientTestPrec
				if math.Abs(actVal) > 1 {
					prec = gradientTestPrec * math.Abs(actVal)
				}
				if math.Abs(expVal-actVal) > prec {
					t.Errorf(prefix+"output %d variable %d: expected %f got %f (idx %d)",
						i, variableIdx, expVal, actVal, j)
				}
			}
		}
	}
}

func approximateJacobian(f func() rnn.BlockOutput, v []*autofunc.Variable) []autofunc.Gradient {
	var res []autofunc.Gradient
	tmp := copyOutput(f())
	for partIdx, list := range [][]linalg.Vector{tmp.Outputs(), tmp.States()} {
		for i, out := range list {
			for j := range out {
				grad := autofunc.Gradient{}
				for _, variable := range v {
					varGrad := make(linalg.Vector, len(variable.Vector))
					for k := range variable.Vector {
						varGrad[k] = dividedDifference(&variable.Vector[k], func() float64 {
							x := f()
							if partIdx == 0 {
								return x.Outputs()[i][j]
							} else {
								return x.States()[i][j]
							}
						})
					}
					grad[variable] = varGrad
				}
				res = append(res, grad)
			}
		}
	}
	return res
}

func approximateRJacobian(f func() rnn.BlockROutput, v []*autofunc.Variable,
	rv autofunc.RVector) []autofunc.RGradient {
	leftGradient := rApproximation(rv, -gradientTestDelta, func() interface{} {
		return approximateJacobian(func() rnn.BlockOutput {
			return &rOutputToOutput{O: f()}
		}, v)
	}).([]autofunc.Gradient)
	rightGradient := rApproximation(rv, gradientTestDelta, func() interface{} {
		return approximateJacobian(func() rnn.BlockOutput {
			return &rOutputToOutput{O: f()}
		}, v)
	}).([]autofunc.Gradient)
	res := make([]autofunc.RGradient, len(leftGradient))
	for i, left := range leftGradient {
		right := rightGradient[i]
		left.Scale(-1)
		left.Add(right)
		left.Scale(1 / (2 * gradientTestDelta))
		res[i] = autofunc.RGradient(left)
	}
	return res
}

func approximateROutput(f func() rnn.BlockOutput, rv autofunc.RVector) (routputs,
	rstates []linalg.Vector) {
	leftOutput := rApproximation(rv, -gradientTestDelta, func() interface{} {
		return copyOutput(f())
	}).(rnn.BlockOutput)
	rightOutput := rApproximation(rv, gradientTestDelta, func() interface{} {
		return copyOutput(f())
	}).(rnn.BlockOutput)
	for i, left := range leftOutput.Outputs() {
		right := rightOutput.Outputs()[i]
		left.Scale(-1).Add(right).Scale(1 / (2 * gradientTestDelta))
		routputs = append(routputs, left)
	}
	for i, left := range leftOutput.States() {
		right := rightOutput.States()[i]
		left.Scale(-1).Add(right).Scale(1 / (2 * gradientTestDelta))
		rstates = append(rstates, left)
	}
	return
}

func rApproximation(rv autofunc.RVector, rAdd float64, f func() interface{}) interface{} {
	variableBackups := map[*autofunc.Variable]linalg.Vector{}
	for variable := range rv {
		variableBackups[variable] = make(linalg.Vector, len(variable.Vector))
		copy(variableBackups[variable], variable.Vector)
	}
	for variable, rvec := range rv {
		variable.Vector.Add(rvec.Copy().Scale(rAdd))
	}
	res := f()
	for variable := range rv {
		copy(variable.Vector, variableBackups[variable])
	}
	return res
}

func exactJacobian(f func() rnn.BlockOutput, v []*autofunc.Variable) []autofunc.Gradient {
	var res []autofunc.Gradient
	tmp := f()
	for partIdx, list := range [][]linalg.Vector{tmp.Outputs(), tmp.States()} {
		for i, out := range list {
			for j := range out {
				grad := autofunc.Gradient{}
				for _, variable := range v {
					grad[variable] = make(linalg.Vector, len(variable.Vector))
				}
				upstreamVec := make([]linalg.Vector, len(list))
				for k := range upstreamVec {
					upstreamVec[k] = make(linalg.Vector, len(list[k]))
				}
				upstreamVec[i][j] = 1
				var upstream *rnn.UpstreamGradient
				if partIdx == 0 {
					upstream = &rnn.UpstreamGradient{Outputs: upstreamVec}
				} else {
					upstream = &rnn.UpstreamGradient{States: upstreamVec}
				}
				tmp.Gradient(upstream, grad)
				res = append(res, grad)
			}
		}
	}
	return res
}

func exactRJacobian(f func() rnn.BlockROutput, v []*autofunc.Variable) []autofunc.RGradient {
	var res []autofunc.RGradient
	tmp := f()
	for partIdx, list := range [][]linalg.Vector{tmp.Outputs(), tmp.States()} {
		for i, out := range list {
			for j := range out {
				grad := autofunc.RGradient{}
				for _, variable := range v {
					grad[variable] = make(linalg.Vector, len(variable.Vector))
				}
				upstreamVec := make([]linalg.Vector, len(list))
				upstreamRVec := make([]linalg.Vector, len(list))
				for k := range upstreamVec {
					upstreamVec[k] = make(linalg.Vector, len(list[k]))
					upstreamRVec[k] = make(linalg.Vector, len(list[k]))
				}
				upstreamVec[i][j] = 1
				var upstream *rnn.UpstreamRGradient
				if partIdx == 0 {
					upstream = &rnn.UpstreamRGradient{
						UpstreamGradient: rnn.UpstreamGradient{Outputs: upstreamVec},
						ROutputs:         upstreamRVec,
					}
				} else {
					upstream = &rnn.UpstreamRGradient{
						UpstreamGradient: rnn.UpstreamGradient{States: upstreamVec},
						RStates:          upstreamRVec,
					}
				}
				tmp.RGradient(upstream, grad, nil)
				res = append(res, grad)
			}
		}
	}
	return res
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

type rOutputToOutput struct {
	O rnn.BlockROutput
}

func (r *rOutputToOutput) Outputs() []linalg.Vector {
	return r.O.Outputs()
}

func (r *rOutputToOutput) States() []linalg.Vector {
	return r.O.States()
}

func (r *rOutputToOutput) Gradient(u *rnn.UpstreamGradient, grad autofunc.Gradient) {
	ur := &rnn.UpstreamRGradient{UpstreamGradient: *u}
	if u.States != nil {
		ur.RStates = make([]linalg.Vector, len(u.States))
		for i := range ur.RStates {
			ur.RStates[i] = make(linalg.Vector, len(u.States[i]))
		}
	}
	if u.Outputs != nil {
		ur.ROutputs = make([]linalg.Vector, len(u.Outputs))
		for i := range ur.ROutputs {
			ur.ROutputs[i] = make(linalg.Vector, len(u.Outputs[i]))
		}
	}
	r.O.RGradient(ur, autofunc.RGradient{}, grad)
}

type copiedOutput struct {
	CopiedOutputs []linalg.Vector
	CopiedStates  []linalg.Vector
}

func copyOutput(o rnn.BlockOutput) *copiedOutput {
	res := &copiedOutput{
		CopiedOutputs: make([]linalg.Vector, len(o.Outputs())),
		CopiedStates:  make([]linalg.Vector, len(o.States())),
	}
	for i, ot := range o.Outputs() {
		res.CopiedOutputs[i] = make(linalg.Vector, len(ot))
		copy(res.CopiedOutputs[i], ot)
	}
	for i, ot := range o.States() {
		res.CopiedStates[i] = make(linalg.Vector, len(ot))
		copy(res.CopiedStates[i], ot)
	}
	return res
}

func (c *copiedOutput) Outputs() []linalg.Vector {
	return c.CopiedOutputs
}

func (c *copiedOutput) States() []linalg.Vector {
	return c.CopiedStates
}

func (c *copiedOutput) Gradient(u *rnn.UpstreamGradient, grad autofunc.Gradient) {
	panic("gradient not implement")
}
