package rnntest

import (
	"math"
	"math/rand"
	"runtime"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

var gradienterTestSamples neuralnet.SliceSampleSet
var gradienterTestBlock rnn.StackedBlock
var gradienterTestCost = neuralnet.CrossEntropyCost{}

const (
	gradienterTestPrec    = 1e-6
	gradienterTestInSize  = 10
	gradienterTestOutSize = 10
)

func init() {
	gradienterTestSamples = neuralnet.SliceSampleSet{
		rnn.Sequence{},
	}
	for i := 0; i < 100; i++ {
		seqLen := rand.Intn(10) + 5
		var seq rnn.Sequence
		for i := 0; i < seqLen; i++ {
			input := make(linalg.Vector, gradienterTestInSize)
			for i := range input {
				input[i] = rand.NormFloat64()
			}
			output := make(linalg.Vector, gradienterTestOutSize)
			for i := range output {
				output[i] = rand.Float64()
			}
			seq.Inputs = append(seq.Inputs, input)
			seq.Outputs = append(seq.Outputs, output)
		}
		gradienterTestSamples = append(gradienterTestSamples, seq)
	}
	outBlock := NewSquareBlock(0)
	l := rnn.NewLSTM(gradienterTestInSize, gradienterTestOutSize)
	gradienterTestBlock = rnn.StackedBlock{l, outBlock}
}

func TestFullRGradienterBasic(t *testing.T) {
	g := &rnn.FullRGradienter{
		Learner:       gradienterTestBlock,
		CostFunc:      gradienterTestCost,
		MaxLanes:      1,
		MaxGoroutines: 1,
	}
	testRGradienter(t, g)
}

func TestFullRGradienterConcurrent(t *testing.T) {
	n := runtime.GOMAXPROCS(0)
	runtime.GOMAXPROCS(10)
	g := &rnn.FullRGradienter{
		Learner:       gradienterTestBlock,
		CostFunc:      gradienterTestCost,
		MaxLanes:      1,
		MaxGoroutines: 10,
	}
	testRGradienter(t, g)
	runtime.GOMAXPROCS(n)
}

func TestFullRGradienterWideLanes(t *testing.T) {
	g := &rnn.FullRGradienter{
		Learner:       gradienterTestBlock,
		CostFunc:      gradienterTestCost,
		MaxLanes:      3,
		MaxGoroutines: 1,
	}
	testRGradienter(t, g)
}

func TestFullRGradienterConcurrentWideLanes(t *testing.T) {
	g := &rnn.FullRGradienter{
		Learner:       gradienterTestBlock,
		CostFunc:      gradienterTestCost,
		MaxLanes:      2,
		MaxGoroutines: 2,
	}
	testRGradienter(t, g)
}

func testRGradienter(t *testing.T, g neuralnet.RGradienter) {
	rv := autofunc.RVector(autofunc.NewGradient(gradienterTestBlock.Parameters()))
	for _, v := range rv {
		for i := range v {
			v[i] = rand.NormFloat64()
		}
	}
	expected, expectedR := expectedRGradient(rv, gradienterTestBlock, gradienterTestCost,
		gradienterTestSamples)
	actual := g.Gradient(gradienterTestSamples)
	compareGrads(t, "gradient", actual, expected)
	actual, actualR := g.RGradient(rv, gradienterTestSamples)
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
	for i := 0; i < samples.Len(); i++ {
		seq := samples.GetSample(i).(rnn.Sequence)
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
