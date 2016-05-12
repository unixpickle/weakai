package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

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
}

// firstBitTest builds a neural network to:
// - output 0 for inputs starting with a 1
// - output 1 for inputs starting with a 0.
func firstBitTest() {
	trainingSamples := make([][]float64, FirstBitTrainingSize)
	trainingOutputs := make([][]float64, FirstBitTrainingSize)
	for i := range trainingSamples {
		trainingSamples[i] = make([]float64, FirstBitInputSize)
		for j := range trainingSamples[i] {
			trainingSamples[i][j] = float64(rand.Intn(2))
		}
		trainingOutputs[i] = []float64{1 - trainingSamples[i][0]}
	}

	network, _ := neuralnet.NewNetwork([]neuralnet.LayerPrototype{
		&neuralnet.DenseParams{
			Activation:  neuralnet.Sigmoid{},
			InputCount:  FirstBitInputSize,
			OutputCount: FirstBitHiddenSize,
		},
		&neuralnet.DenseParams{
			Activation:  neuralnet.Sigmoid{},
			InputCount:  FirstBitHiddenSize,
			OutputCount: 1,
		},
	})

	trainer := neuralnet.SGD{
		CostFunc:         neuralnet.MeanSquaredCost{},
		Inputs:           trainingSamples,
		Outputs:          trainingOutputs,
		StepSize:         0.1,
		StepDecreaseRate: 0,
		Epochs:           100,
	}

	network.Randomize()
	trainer.Train(network)

	var totalError float64
	var maxPossibleError float64
	for i := 0; i < 50; i++ {
		sample := make([]float64, FirstBitInputSize)
		for j := range sample {
			sample[j] = float64(rand.Intn(2))
		}
		network.SetInput(sample)
		network.PropagateForward()
		output := network.Output()[0]
		amountError := math.Abs(output - (1 - sample[0]))
		totalError += amountError
		maxPossibleError += 1.0
	}

	fmt.Printf("firstBitTest() error rate: %f\n", totalError/maxPossibleError)
}

// horizontalLineTest builds a neural network
// to accept bitmaps with horizontal lines.
func horizontalLineTest() {
	trainingSamples := make([][]float64, GridTrainingSize)
	trainingOutputs := make([][]float64, GridTrainingSize)
	for i := range trainingSamples {
		trainingSamples[i] = randomBitmap()
		if bitmapHasHorizontal(trainingSamples[i]) {
			trainingOutputs[i] = []float64{1}
		} else {
			trainingOutputs[i] = []float64{0}
		}
	}

	network, _ := neuralnet.NewNetwork([]neuralnet.LayerPrototype{
		&neuralnet.DenseParams{
			Activation:  neuralnet.Sigmoid{},
			InputCount:  GridInputSide * GridInputSide,
			OutputCount: GridHiddenSize,
		},
		&neuralnet.DenseParams{
			Activation:  neuralnet.Sigmoid{},
			InputCount:  GridHiddenSize,
			OutputCount: 1,
		},
	})

	trainer := neuralnet.SGD{
		CostFunc:         neuralnet.MeanSquaredCost{},
		Inputs:           trainingSamples,
		Outputs:          trainingOutputs,
		StepSize:         0.1,
		StepDecreaseRate: 1e-4,
		Epochs:           1000,
	}

	network.Randomize()
	trainer.Train(network)

	var trainingError float64
	var maxTrainingError float64
	for i, sample := range trainingSamples {
		network.SetInput(sample)
		network.PropagateForward()
		output := network.Output()[0]
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
		network.SetInput(sample)
		network.PropagateForward()
		output := network.Output()[0]
		amountError := math.Abs(output - expected)
		totalError += amountError
		maxPossibleError += 1.0
	}

	fmt.Printf("horizontalLineTest() training error: %f; cross error: %f\n",
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
