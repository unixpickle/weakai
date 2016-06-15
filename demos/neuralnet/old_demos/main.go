package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	FirstBitTrainingSize = 40
	FirstBitInputSize    = 8
	FirstBitHiddenSize   = 3
)

const (
	GridTrainingSize = 3000
	GridInputSide    = 4
	GridHiddenSize   = 10
)

func main() {
	rand.Seed(time.Now().UnixNano())
	firstBitTest()
	horizontalLineTest()
	horizontalLineConvTest()
}

// firstBitTest builds a neural network to:
// - output 0 for inputs starting with a 1
// - output 1 for inputs starting with a 0.
func firstBitTest() {
	trainingSamples := make([]linalg.Vector, FirstBitTrainingSize)
	trainingOutputs := make([]linalg.Vector, FirstBitTrainingSize)
	for i := range trainingSamples {
		trainingSamples[i] = make(linalg.Vector, FirstBitInputSize)
		for j := range trainingSamples[i] {
			trainingSamples[i][j] = float64(rand.Intn(2))
		}
		trainingOutputs[i] = []float64{1 - trainingSamples[i][0]}
	}
	samples := neuralnet.VectorSampleSet(trainingSamples, trainingOutputs)

	network := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  FirstBitInputSize,
			OutputCount: FirstBitHiddenSize,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  FirstBitHiddenSize,
			OutputCount: 1,
		},
		&neuralnet.Sigmoid{},
	}
	network.Randomize()

	batcher := &neuralnet.SingleRGradienter{
		Learner:  network,
		CostFunc: neuralnet.MeanSquaredCost{},
	}
	neuralnet.SGD(batcher, samples, 0.2, 100000, 1)

	var totalError float64
	var maxPossibleError float64
	for i := 0; i < 50; i++ {
		sample := make([]float64, FirstBitInputSize)
		for j := range sample {
			sample[j] = float64(rand.Intn(2))
		}
		result := network.Apply(&autofunc.Variable{sample})
		output := result.Output()[0]
		amountError := math.Abs(output - (1 - sample[0]))
		totalError += amountError
		maxPossibleError += 1.0
	}

	fmt.Printf("firstBitTest() error rate: %f\n", totalError/maxPossibleError)
}

// horizontalLineTest builds a neural network
// to accept bitmaps with horizontal lines.
func horizontalLineTest() {
	network := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  GridInputSide * GridInputSide,
			OutputCount: GridHiddenSize,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  GridHiddenSize,
			OutputCount: 1,
		},
		&neuralnet.Sigmoid{},
	}
	runHorizontalLineTest("horizontalLineTest", network)
}

// horizontalLineConvTest is like
// horizontalLineTest, but it uses
// a convolutional layer.
func horizontalLineConvTest() {
	network := neuralnet.Network{
		&neuralnet.ConvLayer{
			FilterCount:  4,
			FilterWidth:  2,
			FilterHeight: 2,
			Stride:       1,
			InputWidth:   4,
			InputHeight:  4,
			InputDepth:   1,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  3 * 3 * 4,
			OutputCount: 4,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  4,
			OutputCount: 1,
		},
		&neuralnet.Sigmoid{},
	}
	runHorizontalLineTest("horizontalLineConvTest", network)
}

func runHorizontalLineTest(name string, network neuralnet.Network) {
	trainingSamples := make([]linalg.Vector, GridTrainingSize)
	trainingOutputs := make([]linalg.Vector, GridTrainingSize)
	for i := range trainingSamples {
		trainingSamples[i] = randomBitmap()
		if bitmapHasHorizontal(trainingSamples[i]) {
			trainingOutputs[i] = []float64{1}
		} else {
			trainingOutputs[i] = []float64{0}
		}
	}
	samples := neuralnet.VectorSampleSet(trainingSamples, trainingOutputs)

	network.Randomize()
	batcher := &neuralnet.SingleRGradienter{
		Learner:  network,
		CostFunc: neuralnet.MeanSquaredCost{},
	}
	neuralnet.SGD(batcher, samples, 0.1, 1000, 100)

	var trainingError float64
	var maxTrainingError float64
	for i, sample := range trainingSamples {
		result := network.Apply(&autofunc.Variable{sample})
		output := result.Output()[0]
		amountError := math.Abs(output - trainingOutputs[i][0])
		trainingError += amountError
		maxTrainingError += 1.0
	}

	var totalError float64
	var maxPossibleError float64
	for i := 0; i < 50; i++ {
		sample := randomBitmap()
		var expected float64
		if bitmapHasHorizontal(sample) {
			expected = 1
		}
		result := network.Apply(&autofunc.Variable{sample})
		output := result.Output()[0]
		amountError := math.Abs(output - expected)
		totalError += amountError
		maxPossibleError += 1.0
	}

	fmt.Printf("%s() training error: %f; cross error: %f\n", name,
		trainingError/maxTrainingError, totalError/maxPossibleError)
}

func randomBitmap() []float64 {
	res := make([]float64, GridInputSide*GridInputSide)

	for y := 0; y < GridInputSide; y++ {
		zeroX := rand.Intn(GridInputSide)
		for x := 0; x < GridInputSide; x++ {
			if x == zeroX {
				continue
			}
			res[y*GridInputSide+x] = float64(rand.Intn(2))
		}
	}

	hasLine := rand.Intn(2) == 1
	if hasLine {
		line := rand.Intn(GridInputSide)
		for x := 0; x < GridInputSide; x++ {
			res[x+line*GridInputSide] = 1
		}
	}

	return res
}

func bitmapHasHorizontal(bitmap []float64) bool {
	for y := 0; y < GridInputSide; y++ {
		hasNon1 := false
		for x := 0; x < GridInputSide; x++ {
			if bitmap[y*GridInputSide+x] < 0.5 {
				hasNon1 = true
				break
			}
		}
		if !hasNon1 {
			return true
		}
	}
	return false
}
