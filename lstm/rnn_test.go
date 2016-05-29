package lstm

import (
	"fmt"
	"math"
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
)

var rnnTestInputs = []linalg.Vector{
	{0.24845, 0.30611},
	{0.58775, 0.88487},
	{0.56657, 0.31475},
}

var rnnTestOutputs = []linalg.Vector{
	{0.20328, 0.34240},
	{0.14513, 0.61933},
	{0.54129, 0.62811},
}

var rnnTestCostFunc = MeanSquaredCost{}

const (
	rnnTestDelta   = 1e-7
	errorThreshold = 1e-7
)

type rnnTestCase struct {
	inputs     []linalg.Vector
	outputs    []linalg.Vector
	hiddenSize int
	sparse     bool
}

func TestRNNGradientsOneTime(t *testing.T) {
	testRNNGradients(t, &rnnTestCase{
		inputs:     rnnTestInputs[:1],
		outputs:    rnnTestOutputs[:1],
		hiddenSize: 3,
	})
}

func TestRNNGradientsThroughTime(t *testing.T) {
	testRNNGradients(t, &rnnTestCase{
		inputs:     rnnTestInputs,
		outputs:    rnnTestOutputs,
		hiddenSize: 3,
	})
}

func TestRNNGradientsThroughTimeSingleVar(t *testing.T) {
	testRNNGradients(t, &rnnTestCase{
		inputs:     []linalg.Vector{{}, {}},
		outputs:    []linalg.Vector{{0.4976425487191876}, {0}},
		hiddenSize: 1,
		sparse:     true,
	})
}

func testRNNGradients(t *testing.T, r *rnnTestCase) {
	net := rnnForTesting(r)
	_, costGrad := runTestSequence(net, r)

	grad := net.CostGradient(costGrad)

	gradSlices := []linalg.Vector{
		linalg.Vector(grad.OutWeights.Data),
		grad.OutBiases,
		linalg.Vector(grad.OutGate.Data),
		grad.OutGateBiases,
		linalg.Vector(grad.InWeights.Data),
		grad.InBiases,
		linalg.Vector(grad.InGate.Data),
		grad.InGateBiases,
		linalg.Vector(grad.RemGate.Data),
		grad.RemGateBiases,
		linalg.Vector(grad.OutGate.Data),
		grad.OutGateBiases,
	}
	gradSlices = append(gradSlices, grad.Inputs...)

	paramSlices := []linalg.Vector{
		linalg.Vector(net.outWeights.Data),
		net.outBiases,
		linalg.Vector(net.memoryParams.OutGate.Data),
		net.memoryParams.OutGateBiases,
		linalg.Vector(net.memoryParams.InWeights.Data),
		net.memoryParams.InBiases,
		linalg.Vector(net.memoryParams.InGate.Data),
		net.memoryParams.InGateBiases,
		linalg.Vector(net.memoryParams.RemGate.Data),
		net.memoryParams.RemGateBiases,
		linalg.Vector(net.memoryParams.OutGate.Data),
		net.memoryParams.OutGateBiases,
	}
	paramSlices = append(paramSlices, r.inputs...)

	names := []string{"out weight", "out bias", "out gate weight", "out gate bias",
		"in weight", "in bias", "in gate weight", "in gate bias",
		"rem gate weight", "rem gate bias", "out gate weight", "out gate bias"}
	for t := range r.inputs {
		names = append(names, fmt.Sprintf("input at t=%d", t))
	}

	for i, gradSlice := range gradSlices {
		paramSlice := paramSlices[i]
		for j, actual := range gradSlice {
			expected := approxCostDerivative(net, &paramSlice[j], r)
			if math.Abs(actual-expected) > errorThreshold {
				t.Errorf("invalid '%s' partial: got %f expected %f (idx %d)",
					names[i], actual, expected, j)
				break
			}
		}
	}
}

func approxCostDerivative(r *RNN, param *float64, cases *rnnTestCase) float64 {
	old := *param
	*param -= rnnTestDelta
	cost1, _ := runTestSequence(r, cases)
	*param = old + rnnTestDelta
	cost2, _ := runTestSequence(r, cases)
	*param = old
	return (cost2 - cost1) / (2 * rnnTestDelta)
}

func runTestSequence(r *RNN, cases *rnnTestCase) (cost float64, costGrad []linalg.Vector) {
	r.Reset()

	for t, output := range cases.outputs {
		vecOut := r.StepTime(cases.inputs[t], output)
		for i, x := range vecOut {
			diff := x - output[i]
			cost += diff * diff
		}
		grad := make(linalg.Vector, len(vecOut))
		rnnTestCostFunc.Gradient(vecOut, output, grad)
		costGrad = append(costGrad, grad)
	}

	cost /= 2

	return
}

func rnnForTesting(r *rnnTestCase) *RNN {
	inSize := len(r.inputs[0])
	outSize := len(r.outputs[0])
	hiddenSize := r.hiddenSize
	net := NewRNN(inSize, hiddenSize, outSize)
	net.outWeights.Data = []float64{
		0.075439, 0.926433, 0.549735, 0.351469, 0.121239,
		0.415574, 0.094576, 0.727178, 0.858073, 0.758361,
	}[:outSize*(inSize+hiddenSize)]
	if !r.sparse {
		net.outBiases = linalg.Vector{0.44417, 0.81253}[:outSize]
	}

	gateDataLen := hiddenSize * (inSize + hiddenSize)

	net.memoryParams.InWeights.Data = []float64{
		0.062555, 0.903507, 0.986772, 0.375239, 0.610480,
		0.894272, 0.257381, 0.025877, 0.368605, 0.823952,
		0.666342, 0.250000, 0.544260, 0.147394, 0.269642,
	}[:gateDataLen]

	if !r.sparse {
		net.memoryParams.InGate.Data = []float64{
			0.404213, 0.359702, 0.168132, 0.631023, 0.088067,
			0.882302, 0.161765, 0.541852, 0.573962, 0.348698,
			0.681632, 0.609265, 0.853663, 0.462425, 0.709695,
		}[:gateDataLen]
		net.memoryParams.OutGate.Data = []float64{
			0.848029, 0.203344, 0.771436, 0.962591, 0.891356,
			0.023854, 0.462359, 0.778313, 0.580890, 0.625660,
			0.390262, 0.528941, 0.135359, 0.044588, 0.992219,
		}[:gateDataLen]
		net.memoryParams.RemGate.Data = []float64{
			0.108275, 0.493979, 0.542174, 0.414007, 0.400178,
			0.780442, 0.061278, 0.052919, 0.842510, 0.953911,
			0.961888, 0.648135, 0.852180, 0.562492, 0.978203,
		}[:gateDataLen]

		net.memoryParams.InGateBiases = linalg.Vector{0.576392, 0.431650, 0.090567}[:hiddenSize]
		net.memoryParams.OutGateBiases = linalg.Vector{0.52432, 0.25997, 0.59234}[:hiddenSize]
		net.memoryParams.RemGateBiases = linalg.Vector{0.62371, 0.81187, 0.51603}[:hiddenSize]
		net.memoryParams.InBiases = linalg.Vector{0.85035, 0.56379, 0.44774}[:hiddenSize]
	}
	return net
}
