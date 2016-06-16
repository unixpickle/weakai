package rnntest

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

var (
	gradientTestSamples = neuralnet.SampleSet{
		rnn.Sequence{
			Inputs:  []linalg.Vector{{1, 2, 3}, {3, 2, 1}, {1, 3, 2}},
			Outputs: []linalg.Vector{{1, 0.5, 0.3, 0}, {0.2, 0.5, 0.4, 0.9}, {1, 0, 1, 0}},
		},
		/*rnn.Sequence{
			Inputs:  []linalg.Vector{},
			Outputs: []linalg.Vector{},
		},
		rnn.Sequence{
			Inputs:  []linalg.Vector{{3, 2, 1}},
			Outputs: []linalg.Vector{{1, 0, 0.5, 0.2}},
		},*/
	}

	gradientTestBlock = rnn.StackedBlock{rnn.NewLSTM(3, 4), NewSquareBlock(2)}
	gradientTestCost  = neuralnet.MeanSquaredCost{}
)

const gradienterTestPrec = 1e-6

func TestFullRGradienterBasic(t *testing.T) {
	g := &rnn.FullRGradienter{
		Learner:       gradientTestBlock,
		CostFunc:      gradientTestCost,
		MaxLanes:      1,
		MaxGoroutines: 1,
	}
	testRGradienter(t, g)
}

func TestFullRGradienterConcurrent(t *testing.T) {
	g := &rnn.FullRGradienter{
		Learner:       gradientTestBlock,
		CostFunc:      gradientTestCost,
		MaxLanes:      1,
		MaxGoroutines: 10,
	}
	testRGradienter(t, g)
}

func TestFullRGradienterWideLanes(t *testing.T) {
	g := &rnn.FullRGradienter{
		Learner:       gradientTestBlock,
		CostFunc:      gradientTestCost,
		MaxLanes:      3,
		MaxGoroutines: 1,
	}
	testRGradienter(t, g)
}

func TestFullRGradienterConcurrentWideLanes(t *testing.T) {
	g := &rnn.FullRGradienter{
		Learner:       gradientTestBlock,
		CostFunc:      gradientTestCost,
		MaxLanes:      2,
		MaxGoroutines: 2,
	}
	testRGradienter(t, g)
}

func testRGradienter(t *testing.T, g neuralnet.RGradienter) {
	rv := autofunc.RVector(autofunc.NewGradient(gradientTestBlock.Parameters()))
	for _, v := range rv {
		for i := range v {
			v[i] = rand.NormFloat64()
		}
	}
	expected, expectedR := expectedRGradient(rv, gradientTestBlock, gradientTestCost,
		gradientTestSamples)
	actual := g.Gradient(gradientTestSamples)
	compareGrads(t, "gradient", actual, expected)
	actual, actualR := g.RGradient(rv, gradientTestSamples)
	compareGrads(t, "gradient (RGradient)", actual, expected)
	compareGrads(t, "r-gradient", actualR, expectedR)
}

func expectedRGradient(v autofunc.RVector, bl rnn.BlockLearner, cost neuralnet.CostFunc,
	samples neuralnet.SampleSet) (autofunc.Gradient, autofunc.RGradient) {
	if v == nil {
		v = autofunc.RVector{}
	}
	res := autofunc.NewGradient(bl.Parameters())
	resR := autofunc.NewRGradient(bl.Parameters())
	for _, sample := range samples {
		seq := sample.(rnn.Sequence)
		if len(seq.Inputs) == 0 {
			continue
		}
		inState := make(linalg.Vector, bl.StateSize())
		recursiveGrad(v, bl, cost, res, resR, seq, inState, inState)
	}
	return res, resR
}

func recursiveGrad(v autofunc.RVector, bl rnn.BlockLearner, cost neuralnet.CostFunc,
	g autofunc.Gradient, rg autofunc.RGradient, seq rnn.Sequence,
	inState, inRState linalg.Vector) (linalg.Vector, linalg.Vector) {
	input := &rnn.BlockRInput{
		States: []*autofunc.RVariable{&autofunc.RVariable{
			Variable:   &autofunc.Variable{Vector: inState},
			ROutputVec: inRState,
		}},
		Inputs: []*autofunc.RVariable{&autofunc.RVariable{
			Variable:   &autofunc.Variable{Vector: seq.Inputs[0]},
			ROutputVec: make(linalg.Vector, len(seq.Inputs[0])),
		}},
	}
	output := bl.BatchR(v, input)

	outputRVar := &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: output.Outputs()[0]},
		ROutputVec: output.ROutputs()[0],
	}
	costRes := cost.CostR(v, seq.Outputs[0], outputRVar)
	outputGrad := make(linalg.Vector, len(output.Outputs()[0]))
	outputRGrad := make(linalg.Vector, len(outputGrad))
	costRes.PropagateRGradient(linalg.Vector{1}, linalg.Vector{0},
		autofunc.RGradient{outputRVar.Variable: outputRGrad},
		autofunc.Gradient{outputRVar.Variable: outputGrad})

	upstream := &rnn.UpstreamRGradient{}
	upstream.Outputs = append(upstream.Outputs, outputGrad)
	upstream.ROutputs = append(upstream.ROutputs, outputRGrad)

	if len(seq.Inputs) > 1 {
		seq = rnn.Sequence{Inputs: seq.Inputs[1:], Outputs: seq.Outputs[1:]}
		ups, upsR := recursiveGrad(v, bl, cost, g, rg, seq, output.States()[0],
			output.RStates()[0])
		upstream.States = append(upstream.States, ups)
		upstream.RStates = append(upstream.RStates, upsR)
	}

	downstream := make(linalg.Vector, bl.StateSize())
	downstreamR := make(linalg.Vector, bl.StateSize())
	g[input.States[0].Variable] = downstream
	rg[input.States[0].Variable] = downstreamR

	output.RGradient(upstream, rg, g)

	delete(g, input.States[0].Variable)
	delete(rg, input.States[0].Variable)
	return downstream, downstreamR
}

func compareGrads(t *testing.T, prefix string, actual,
	expected map[*autofunc.Variable]linalg.Vector) {
	if len(actual) != len(expected) {
		t.Errorf("%s: expected map length %d got length %d", prefix,
			len(expected), len(actual))
		return
	}
	for variable, expectedVec := range expected {
		actualVec := actual[variable]
		if len(expectedVec) != len(actualVec) {
			t.Errorf("%s: expected vec length %d got length %d", prefix,
				len(expectedVec), len(actualVec))
			return
		}
		for i, x := range expectedVec {
			a := actualVec[i]
			if math.Abs(a-x) > gradienterTestPrec {
				t.Errorf("%s: got %f expected %f", prefix, a, x)
				return
			}
		}
	}
}
