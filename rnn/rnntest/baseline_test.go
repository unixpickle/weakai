package rnntest

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	BaselineSeqCount  = 5
	BaselineSeqMinLen = 1
	BaselineSeqMaxLen = 5
)

// TestBaselineOutput makes sure that the BatcherBlock +
// BlockSeqFunc combo produces the right output, since
// that combo will be used for the rest of the tests.
func TestBaselineOutput(t *testing.T) {
	network := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  4,
			OutputCount: 6,
		},
		neuralnet.HyperbolicTangent{},
	}
	network.Randomize()

	for stateSize := 0; stateSize < 4; stateSize++ {
		start := &autofunc.Variable{Vector: make(linalg.Vector, stateSize)}
		for i := range start.Vector {
			start.Vector[i] = rand.NormFloat64()
		}
		toTest := rnn.BlockSeqFunc{
			B: &rnn.BatcherBlock{
				B:         network.BatchLearner(),
				StateSize: stateSize,
				Start:     start,
			},
		}
		seqs, rv := randBaselineTestSeqs(network, 4-stateSize)
		rv[start] = make(linalg.Vector, len(start.Vector))
		for i := range rv[start] {
			rv[start][i] = rand.NormFloat64()
		}
		res := toTest.ApplySeqsR(rv, seqfunc.VarRResult(rv, seqs))
		actual := res.OutputSeqs()
		actualR := res.ROutputSeqs()
		expected, expectedR := manualNetworkSeq(rv, network, start, seqs, stateSize)
		if len(expected) != len(actual) {
			t.Errorf("stateSize %d: len(expected) [%d] != len(actual) [%d]", stateSize,
				len(expected), len(actual))
			continue
		}
		for i, act := range actual {
			actR := actualR[i]
			exp := expected[i]
			expR := expectedR[i]
			if len(act) != len(exp) {
				t.Errorf("stateSize %d seq %d: len(act) [%d] != len(exp) [%d]",
					stateSize, i, len(act), len(act))
				continue
			}
			for j, a := range act {
				x := exp[j]
				if len(a) != len(x) || x.Copy().Scale(-1).Add(a).MaxAbs() > 1e-5 {
					t.Errorf("stateSize %d seq %d entry %d: expected %v got %v",
						stateSize, i, j, x, a)
				}
			}
			for j, a := range actR {
				x := expR[j]
				if len(a) != len(x) || x.Copy().Scale(-1).Add(a).MaxAbs() > 1e-5 {
					t.Errorf("stateSize %d seq %d entry %d (R): expected %v got %v",
						stateSize, i, j, x, a)
				}
			}
		}
	}
}

func TestBaselineChecks(t *testing.T) {
	network := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  4,
			OutputCount: 6,
		},
		neuralnet.HyperbolicTangent{},
	}
	network.Randomize()

	for stateSize := 0; stateSize < 4; stateSize++ {
		start := &autofunc.Variable{Vector: make(linalg.Vector, stateSize)}
		for i := range start.Vector {
			start.Vector[i] = rand.NormFloat64()
		}
		toTest := &rnn.BlockSeqFunc{
			B: &rnn.BatcherBlock{
				B:         network.BatchLearner(),
				StateSize: stateSize,
				Start:     start,
			},
		}
		seqs, rv := randBaselineTestSeqs(network, 4-stateSize)
		rv[start] = make(linalg.Vector, len(start.Vector))
		for i := range rv[start] {
			rv[start][i] = rand.NormFloat64()
		}
		vars := make([]*autofunc.Variable, 0, len(rv))
		for v := range rv {
			vars = append(vars, v)
		}
		checker := &functest.SeqRFuncChecker{
			F:     toTest,
			Vars:  vars,
			Input: seqs,
			RV:    rv,
		}
		checker.FullCheck(t)
	}
}

func randBaselineTestSeqs(l sgd.Learner, inSize int) ([][]*autofunc.Variable, autofunc.RVector) {
	var seqs [][]*autofunc.Variable

	// Empty sequences will help test for certain edge cases.
	seqs = append(seqs, nil)

	rv := autofunc.RVector{}
	for i := 0; i < BaselineSeqCount; i++ {
		count := rand.Intn(BaselineSeqMaxLen-BaselineSeqMinLen) + BaselineSeqMinLen
		seq := make([]*autofunc.Variable, count)
		for j := range seq {
			seq[j] = &autofunc.Variable{
				Vector: make(linalg.Vector, inSize),
			}
			for k := range seq[j].Vector {
				seq[j].Vector[k] = rand.NormFloat64()
			}
			rv[seq[j]] = make(linalg.Vector, inSize)
			for k := range rv[seq[j]] {
				rv[seq[j]][k] = rand.NormFloat64()
			}
		}
		seqs = append(seqs, seq)
	}

	// Empty sequences will help test for certain edge cases.
	seqs = append(seqs, nil)

	for _, param := range l.Parameters() {
		rv[param] = make(linalg.Vector, len(param.Vector))
		for i := range rv[param] {
			rv[param][i] = rand.NormFloat64()
		}
	}

	return seqs, rv
}

func manualNetworkSeq(rv autofunc.RVector, f autofunc.RFunc, start *autofunc.Variable,
	ins [][]*autofunc.Variable, stateSize int) (out, outR [][]linalg.Vector) {
	out = make([][]linalg.Vector, len(ins))
	outR = make([][]linalg.Vector, len(ins))
	for seqIdx, inSeq := range ins {
		var state autofunc.RResult = autofunc.NewRVariable(start, rv)
		for _, in := range inSeq {
			inR := rv[in]

			packedIn := append(linalg.Vector{}, in.Output()...)
			packedIn = append(packedIn, state.Output()...)
			packedInR := append(linalg.Vector{}, inR...)
			packedInR = append(packedInR, state.ROutput()...)

			stepOut := f.ApplyR(rv, &autofunc.RVariable{
				Variable:   &autofunc.Variable{Vector: packedIn},
				ROutputVec: packedInR,
			})
			outSize := len(stepOut.Output()) - stateSize
			out[seqIdx] = append(out[seqIdx], stepOut.Output()[:outSize])
			outR[seqIdx] = append(outR[seqIdx], stepOut.ROutput()[:outSize])
			state = &autofunc.RVariable{
				Variable:   &autofunc.Variable{Vector: stepOut.Output()[outSize:]},
				ROutputVec: stepOut.ROutput()[outSize:],
			}
		}
	}
	return
}
