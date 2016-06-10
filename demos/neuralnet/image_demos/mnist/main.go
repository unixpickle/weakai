package main

import (
	"log"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/mnist"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	HiddenSize = 300
	LabelCount = 10
	StepSize   = 1e-2
	BatchSize  = 20

	FilterSize   = 3
	FilterCount  = 5
	FilterStride = 1

	MaxPoolingSpan = 3
)

func main() {
	training := mnist.LoadTrainingDataSet()
	crossValidation := mnist.LoadTestingDataSet()

	net := createNet(training)

	trainingSamples := dataSetSamples(training)
	gradienter := &neuralnet.BatchRGradienter{
		Learner:  net,
		CostFunc: neuralnet.MeanSquaredCost{},
	}

	neuralnet.SGDInteractive(gradienter, trainingSamples, StepSize, BatchSize, func() bool {
		log.Println("Printing score...")
		printScore("Cross", net, crossValidation)
		log.Println("Running training round...")
		return true
	})
}

func createNet(d mnist.DataSet) neuralnet.Network {
	convOutWidth := (d.Width-FilterSize)/FilterStride + 1
	convOutHeight := (d.Height-FilterSize)/FilterStride + 1

	poolOutWidth := convOutWidth / MaxPoolingSpan
	if convOutWidth%MaxPoolingSpan != 0 {
		poolOutWidth++
	}
	poolOutHeight := convOutWidth / MaxPoolingSpan
	if convOutHeight%MaxPoolingSpan != 0 {
		poolOutHeight++
	}

	net := neuralnet.Network{
		&neuralnet.ConvLayer{
			FilterCount:  FilterCount,
			FilterWidth:  FilterSize,
			FilterHeight: FilterSize,
			Stride:       FilterStride,
			InputWidth:   d.Width,
			InputHeight:  d.Height,
			InputDepth:   1,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.MaxPoolingLayer{
			XSpan:       MaxPoolingSpan,
			YSpan:       MaxPoolingSpan,
			InputWidth:  convOutWidth,
			InputHeight: convOutHeight,
			InputDepth:  FilterCount,
		},
		&neuralnet.DenseLayer{
			InputCount:  poolOutWidth * poolOutHeight * FilterCount,
			OutputCount: HiddenSize,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  HiddenSize,
			OutputCount: LabelCount,
		},
		&neuralnet.Sigmoid{},
	}
	net.Randomize()

	return net
}

func printScore(prefix string, n neuralnet.Network, d mnist.DataSet) {
	classifier := func(v []float64) int {
		result := n.Apply(&autofunc.Variable{v})
		defer result.Release()
		return outputIdx(result)
	}
	correctCount := d.NumCorrect(classifier)
	histogram := d.CorrectnessHistogram(classifier)
	log.Printf("%s: %d/%d - %s", prefix, correctCount, len(d.Samples), histogram)
}

func outputIdx(r autofunc.Result) int {
	out := r.Output()
	var maxIdx int
	var max float64
	for i, x := range out {
		if i == 0 || x > max {
			max = x
			maxIdx = i
		}
	}
	return maxIdx
}

func dataSetSamples(d mnist.DataSet) *neuralnet.SampleSet {
	labelVecs := d.LabelVectors()
	inputVecs := d.IntensityVectors()
	return &neuralnet.SampleSet{
		Inputs:  vecVec(inputVecs),
		Outputs: vecVec(labelVecs),
	}
}

func vecVec(f [][]float64) []linalg.Vector {
	res := make([]linalg.Vector, len(f))
	for i, x := range f {
		res[i] = x
	}
	return res
}
