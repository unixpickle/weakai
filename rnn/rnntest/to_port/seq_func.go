package rnntest

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

const (
	seqFuncTestDelta = 1e-5
	seqFuncTestPrec  = 1e-5
)

// SeqFuncTest performs gradient checking on an
// rnn.SeqFunc and tests for consistency between
// BatchSeqs and BatchSeqsR.
type SeqFuncTest struct {
	S        rnn.SeqFunc
	Params   []*autofunc.Variable
	TestSeqs [][]linalg.Vector
}

func (s *SeqFuncTest) Run(t *testing.T) {
	expectedOut := s.actualOutput()

	rv := autofunc.RVector{}
	for _, p := range s.Params {
		rv[p] = make(linalg.Vector, len(p.Vector))
		for i := range rv[p] {
			rv[p][i] = rand.NormFloat64()
		}
	}
	rSeqs := seqsToRVarSeqs(s.TestSeqs)
	s.augmentRVector(rv, rSeqs)

	upstream := randomUpstreamSeqs(s.TestSeqs, seqTimestepDimension(expectedOut))
	upstreamR := randomUpstreamSeqs(s.TestSeqs, seqTimestepDimension(expectedOut))

	actualOut, actualROut := s.actualOutputR(rv, rSeqs)
	expectedROut := s.expectedROutput(rv, rSeqs)
	testSequencesEqual(t, "output (R)", actualOut, expectedOut)
	testSequencesEqual(t, "r-output", actualROut, expectedROut)

	actualRGrad, actualGrad := s.actualGradients(rv, rSeqs, upstream, upstreamR)
	expectedRGrad, expectedGrad := s.expectedGradients(rv, rSeqs, upstream, upstreamR)
	testGradMapsEqual(t, "gradients (R)", actualGrad, expectedGrad)
	testGradMapsEqual(t, "r-gradients", actualRGrad, expectedRGrad)

	actualGrad = s.actualGradientNonR(rv, rSeqs, upstream)
	testGradMapsEqual(t, "gradients", actualGrad, expectedGrad)
}

func (s *SeqFuncTest) actualOutput() [][]linalg.Vector {
	return s.S.BatchSeqs(seqsToVarSeqs(s.TestSeqs)).OutputSeqs()
}

func (s *SeqFuncTest) actualOutputR(rv autofunc.RVector, rSeqs [][]autofunc.RResult) (output,
	outputR [][]linalg.Vector) {
	res := s.S.BatchSeqsR(rv, rSeqs)
	return res.OutputSeqs(), res.ROutputSeqs()
}

func (s *SeqFuncTest) actualGradientNonR(rv autofunc.RVector,
	rSeqs [][]autofunc.RResult,
	upstream [][]linalg.Vector) autofunc.Gradient {
	params := make([]*autofunc.Variable, 0, len(rv))
	for p := range rv {
		params = append(params, p)
	}

	g := autofunc.NewGradient(params)

	var seqs [][]autofunc.Result
	for _, rSeq := range rSeqs {
		var seq []autofunc.Result
		for _, x := range rSeq {
			seq = append(seq, x.(*autofunc.RVariable).Variable)
		}
		seqs = append(seqs, seq)
	}

	output := s.S.BatchSeqs(seqs)
	output.Gradient(upstream, g)

	return g
}

func (s *SeqFuncTest) actualGradients(rv autofunc.RVector,
	rSeqs [][]autofunc.RResult, upstream,
	upstreamR [][]linalg.Vector) (rg autofunc.RGradient, g autofunc.Gradient) {
	params := make([]*autofunc.Variable, 0, len(rv))
	for p := range rv {
		params = append(params, p)
	}

	g = autofunc.NewGradient(params)
	rg = autofunc.NewRGradient(params)

	output := s.S.BatchSeqsR(rv, rSeqs)
	output.RGradient(upstream, upstreamR, rg, g)

	return
}

func (s *SeqFuncTest) expectedROutput(rv autofunc.RVector,
	rSeqs [][]autofunc.RResult) [][]linalg.Vector {
	var leftOutput, rightOutput [][]linalg.Vector
	s.runWithR(rv, -seqFuncTestDelta, func() {
		leftOutput = s.actualOutput()
	})
	s.runWithR(rv, seqFuncTestDelta, func() {
		rightOutput = s.actualOutput()
	})
	outputR := make([][]linalg.Vector, len(rSeqs))
	for i, rightSeq := range rightOutput {
		leftSeq := leftOutput[i]
		var outVec []linalg.Vector
		for j, right := range rightSeq {
			left := leftSeq[j]
			diff := left.Copy().Scale(-1).Add(right).Scale(0.5 / seqFuncTestDelta)
			outVec = append(outVec, diff)
		}
		outputR[i] = outVec
	}
	return outputR
}

func (s *SeqFuncTest) expectedGradients(rv autofunc.RVector,
	rSeqs [][]autofunc.RResult, upstream,
	upstreamR [][]linalg.Vector) (rg autofunc.RGradient, g autofunc.Gradient) {
	params := make([]*autofunc.Variable, 0, len(rv))
	for p := range rv {
		params = append(params, p)
	}

	g = autofunc.NewGradient(params)
	rg = autofunc.NewRGradient(params)

	augmentedRV := autofunc.RVector{}
	for x, y := range rv {
		augmentedRV[x] = y
	}
	for i, upstreamSeq := range upstream {
		for j, upstreamVec := range upstreamSeq {
			variable := &autofunc.Variable{Vector: upstreamVec}
			augmentedRV[variable] = upstreamR[i][j]
		}
	}

	for _, variable := range params {
		for i := range variable.Vector {
			paramPtr := &variable.Vector[i]
			gradVal := s.partial(upstream, paramPtr)
			var leftVal, rightVal float64
			s.runWithR(augmentedRV, -seqFuncTestDelta, func() {
				leftVal = s.partial(upstream, paramPtr)
			})
			s.runWithR(augmentedRV, seqFuncTestDelta, func() {
				rightVal = s.partial(upstream, paramPtr)
			})
			g[variable][i] = gradVal
			rg[variable][i] = (rightVal - leftVal) / (2 * seqFuncTestDelta)
		}
	}

	return
}

// runWithR temporarily shifts the variables in an rVec
// by a scaled version of the rVec, runs the function,
// and then unshifts the variables.
func (s *SeqFuncTest) runWithR(rVec autofunc.RVector, scaler float64, f func()) {
	backups := map[*autofunc.Variable]linalg.Vector{}
	for variable, vec := range rVec {
		backup := make(linalg.Vector, len(vec))
		copy(backup, variable.Vector)
		backups[variable] = backup
		variable.Vector.Add(vec.Copy().Scale(scaler))
	}
	f()
	for variable, backup := range backups {
		copy(variable.Vector, backup)
	}
}

// partial approximates the partial derivative of some
// objective function with respect to a parameter.
func (s *SeqFuncTest) partial(upstream [][]linalg.Vector, param *float64) float64 {
	old := *param
	*param = old - seqFuncTestDelta
	out1 := s.actualOutput()
	*param = old + seqFuncTestDelta
	out2 := s.actualOutput()
	*param = old

	var grad float64
	for i, rightSeq := range out2 {
		leftSeq := out1[i]
		for j, right := range rightSeq {
			left := leftSeq[j]
			gradVec := upstream[i][j]
			diff := left.Copy().Scale(-1).Add(right).Scale(0.5 / seqFuncTestDelta)
			grad += diff.Dot(gradVec)
		}
	}
	return grad
}

// augmentRVector makes sure the variables in rSeq are
// included in the RVector.
func (s *SeqFuncTest) augmentRVector(rv autofunc.RVector, rSeq [][]autofunc.RResult) {
	for _, seq := range rSeq {
		for _, rres := range seq {
			rvar := rres.(*autofunc.RVariable)
			rv[rvar.Variable] = rvar.ROutputVec
		}
	}
}

func testSequencesEqual(t *testing.T, label string, actual, expected [][]linalg.Vector) {
	if len(actual) != len(expected) {
		t.Errorf("%s: expected %d outputs but got %d", label, len(expected), len(actual))
		return
	}
	for i, xs := range expected {
		as := actual[i]
		if len(xs) != len(as) {
			t.Errorf("%s: output %d: expected %d timesteps but got %d",
				label, i, len(xs), len(as))
			continue
		}
		for time, xVec := range xs {
			aVec := as[time]
			if len(xVec) != len(aVec) {
				t.Errorf("%s: output %d time %d: expected len %d got %d",
					label, i, time, len(xVec), len(aVec))
			} else {
				for j, x := range xVec {
					a := aVec[j]
					if math.Abs(a-x) > seqFuncTestPrec {
						t.Errorf("%s: output %d time %d entry %d: expected %f got %f",
							label, i, time, j, x, a)
					}
				}
			}
		}
	}
}

func testGradMapsEqual(t *testing.T, label string, actual,
	expected map[*autofunc.Variable]linalg.Vector) {
	if len(actual) != len(expected) {
		t.Errorf("%s: expected len %d got len %d", label, len(expected), len(actual))
		return
	}
	for variable, expectedVec := range expected {
		actualVec := actual[variable]
		if len(actualVec) != len(expectedVec) {
			t.Errorf("%s: variable len should be %d but got %d",
				label, len(expectedVec), len(actualVec))
			continue
		}
		for i, x := range expectedVec {
			a := actualVec[i]
			if math.Abs(a-x) > seqFuncTestPrec {
				t.Errorf("%s: entry %d: should be %f but got %f", label, i, x, a)
			}
		}
	}
}

// seqsToVarSeqs turns a batch of sequences into a batch
// of autofunc.Results by wrapping vectors in variables.
func seqsToVarSeqs(s [][]linalg.Vector) [][]autofunc.Result {
	res := make([][]autofunc.Result, len(s))
	for i, x := range s {
		for _, v := range x {
			res[i] = append(res[i], &autofunc.Variable{Vector: v})
		}
	}
	return res
}

// seqsToRVarSeqs is like seqsToVarSeqs, but it creates
// r-variables with random r-vectors.
func seqsToRVarSeqs(s [][]linalg.Vector) [][]autofunc.RResult {
	rand.Seed(123)
	res := make([][]autofunc.RResult, len(s))
	for i, x := range s {
		for _, v := range x {
			variable := &autofunc.Variable{Vector: v}
			rVec := make(linalg.Vector, len(v))
			for i := range rVec {
				rVec[i] = rand.NormFloat64()
			}
			rVar := &autofunc.RVariable{Variable: variable, ROutputVec: rVec}
			res[i] = append(res[i], rVar)
		}
	}
	return res
}

// randomUpstreamSeqs creates a random upstream gradient.
func randomUpstreamSeqs(s [][]linalg.Vector, outputSize int) [][]linalg.Vector {
	rand.Seed(123123)
	var res [][]linalg.Vector
	for _, v := range s {
		randSeq := make([]linalg.Vector, len(v))
		for i := range v {
			randSeq[i] = make(linalg.Vector, outputSize)
			for j := range randSeq[i] {
				randSeq[i][j] = rand.NormFloat64()
			}
		}
		res = append(res, randSeq)
	}
	return res
}

// seqTimestepDimension returns the size of the first
// entry in the batch of sequences.
func seqTimestepDimension(seqs [][]linalg.Vector) int {
	for _, outSeq := range seqs {
		if len(outSeq) > 0 {
			return len(outSeq[0])
		}
	}
	return 0
}
