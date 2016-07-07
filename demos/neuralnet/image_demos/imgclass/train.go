package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"sort"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	FilterCount  = 10
	FilterCount1 = 10
	HiddenSize   = 60

	ValidationFraction = 0.15
	StepSize           = 0.001
	BatchSize          = 100
	Regularization     = 1e-2
)

func TrainCmd(netPath, dirPath string) {
	log.Println("Loading samples...")
	images, width, height, err := LoadTrainingImages(dirPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	log.Println("Creating network...")

	var network neuralnet.Network
	networkData, err := ioutil.ReadFile(netPath)
	if err == nil {
		network, err = neuralnet.DeserializeNetwork(networkData)
		if err != nil {
			fmt.Fprintln(os.Stderr, "Failed to load network:", err)
			os.Exit(1)
		}
		log.Println("Loaded network from file.")
	} else {
		mean, stddev := sampleStatistics(images)
		convLayer := &neuralnet.ConvLayer{
			FilterCount:  FilterCount,
			FilterWidth:  4,
			FilterHeight: 4,
			Stride:       2,

			InputWidth:  width,
			InputHeight: height,
			InputDepth:  ImageDepth,
		}
		maxLayer := &neuralnet.MaxPoolingLayer{
			XSpan:       3,
			YSpan:       3,
			InputWidth:  convLayer.OutputWidth(),
			InputHeight: convLayer.OutputHeight(),
			InputDepth:  convLayer.OutputDepth(),
		}
		convLayer1 := &neuralnet.ConvLayer{
			FilterCount:  FilterCount1,
			FilterWidth:  3,
			FilterHeight: 3,
			Stride:       2,

			InputWidth:  maxLayer.OutputWidth(),
			InputHeight: maxLayer.OutputHeight(),
			InputDepth:  maxLayer.InputDepth,
		}
		network = neuralnet.Network{
			&neuralnet.RescaleLayer{
				Bias:  -mean,
				Scale: 1 / stddev,
			},
			convLayer,
			neuralnet.HyperbolicTangent{},
			maxLayer,
			neuralnet.HyperbolicTangent{},
			convLayer1,
			neuralnet.HyperbolicTangent{},
			&neuralnet.DenseLayer{
				InputCount: convLayer1.OutputWidth() * convLayer1.OutputHeight() *
					convLayer1.OutputDepth(),
				OutputCount: HiddenSize,
			},
			neuralnet.HyperbolicTangent{},
			&neuralnet.DenseLayer{
				InputCount:  HiddenSize,
				OutputCount: len(images),
			},
			&neuralnet.LogSoftmaxLayer{},
		}
		network.Randomize()
		log.Println("Created new network.")
	}

	samples := neuralSamples(images)
	sgd.ShuffleSampleSet(samples)

	validationCount := int(ValidationFraction * float64(samples.Len()))
	validationSamples := samples.Subset(0, validationCount)
	trainingSamples := samples.Subset(validationCount, samples.Len())

	costFunc := neuralnet.DotCost{}
	gradienter := &sgd.Adam{
		Gradienter: &neuralnet.BatchRGradienter{
			Learner: network.BatchLearner(),
			CostFunc: &neuralnet.RegularizingCost{
				Variables: network.Parameters(),
				Penalty:   Regularization,
				CostFunc:  costFunc,
			},
		},
	}
	sgd.SGDInteractive(gradienter, trainingSamples, StepSize, BatchSize, func() bool {
		log.Printf("Costs: validation=%d/%d cost=%f",
			countCorrect(network, validationSamples), validationSamples.Len(),
			neuralnet.TotalCost(costFunc, network, trainingSamples))
		return true
	})

	data, _ := network.Serialize()
	if err := ioutil.WriteFile(netPath, data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, "Failed to save:", err)
		os.Exit(1)
	}
}

func neuralSamples(m map[string][]linalg.Vector) sgd.SampleSet {
	classes := sortedClasses(m)
	var res sgd.SliceSampleSet
	for i, class := range classes {
		for _, image := range m[class] {
			sample := neuralnet.VectorSample{
				Input:  image,
				Output: make(linalg.Vector, len(classes)),
			}
			sample.Output[i] = 1
			res = append(res, sample)
		}
	}
	return res
}

func sampleStatistics(m map[string][]linalg.Vector) (mean, stddev float64) {
	var count int
	for _, list := range m {
		for _, img := range list {
			for _, x := range img {
				mean += x
				count++
			}
		}
	}
	mean /= float64(count)
	for _, list := range m {
		for _, img := range list {
			for _, x := range img {
				stddev += (x - mean) * (x - mean)
			}
		}
	}
	stddev /= float64(count)
	stddev = math.Sqrt(stddev)
	return
}

func sortedClasses(m map[string][]linalg.Vector) []string {
	keys := make([]string, 0, len(m))
	for key := range m {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func countCorrect(n neuralnet.Network, s sgd.SampleSet) int {
	var count int
	for i := 0; i < s.Len(); i++ {
		sample := s.GetSample(i).(neuralnet.VectorSample)
		output := n.Apply(&autofunc.Variable{Vector: sample.Input}).Output()
		var maxIdx int
		var maxVal float64
		for j, x := range output {
			if x > maxVal || j == 0 {
				maxIdx = j
				maxVal = x
			}
		}
		if sample.Output[maxIdx] == 1 {
			count++
		}
	}
	return count
}
