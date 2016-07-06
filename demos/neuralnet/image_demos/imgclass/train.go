package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"sort"
	"strconv"

	"github.com/unixpickle/hessfree"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	HiddenSize1 = 300
	HiddenSize2 = 100

	TrainSGDEnvVar     = "IMGCLASS_USE_SGD"
	TrainDampingEnvVar = "IMGCLASS_DAMPING"
	ValidationFraction = 0.1
	StepSize           = 0.001
	BatchSize          = 100
	Regularization     = 5e-4
)

func TrainCmd(netPath, dirPath string) {
	log.Println("Loading samples...")
	images, err := LoadTrainingImages(dirPath)
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
		network = neuralnet.Network{
			&neuralnet.RescaleLayer{
				Bias:  -mean,
				Scale: 1 / stddev,
			},
			&neuralnet.DenseLayer{
				InputCount:  ImageSize * ImageSize * ImageDepth,
				OutputCount: HiddenSize1,
			},
			neuralnet.HyperbolicTangent{},
			&neuralnet.DenseLayer{
				InputCount:  HiddenSize1,
				OutputCount: HiddenSize2,
			},
			neuralnet.HyperbolicTangent{},
			&neuralnet.DenseLayer{
				InputCount:  HiddenSize2,
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
	if os.Getenv(TrainSGDEnvVar) != "" {
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
			log.Printf("Costs: validation=%f training=%f",
				neuralnet.TotalCost(costFunc, network, validationSamples),
				neuralnet.TotalCost(costFunc, network, trainingSamples))
			return true
		})
	} else {
		ui := &crossValidationUI{
			UI:                hessfree.NewConsoleUI(),
			costFunc:          costFunc,
			network:           network,
			validationSamples: validationSamples,
		}
		initDamping := 0.3
		if dampingStr := os.Getenv(TrainDampingEnvVar); dampingStr != "" {
			initDamping, err = strconv.ParseFloat(dampingStr, 64)
			if err != nil {
				fmt.Fprintln(os.Stderr, "Invalid damping:", dampingStr)
				os.Exit(1)
			}
			log.Println("Using initial damping", initDamping)
		}
		learner := &hessfree.DampingLearner{
			WrappedLearner: &hessfree.DampingLearner{
				WrappedLearner: &hessfree.NeuralNetLearner{
					Layers:         network[:len(network)-1],
					Output:         network[len(network)-1:],
					Cost:           costFunc,
					MaxConcurrency: 2,
				},
				DampingCoeff: Regularization,
				ChangeRatio:  1,
			},
			DampingCoeff: initDamping,
			UseQuadMin:   true,
			UI:           ui,
		}
		trainer := hessfree.Trainer{
			Learner:   learner,
			Samples:   trainingSamples,
			BatchSize: trainingSamples.Len(),
			UI:        ui,
			Convergence: hessfree.ConvergenceCriteria{
				MinK: 5,
			},
		}
		trainer.Train()
	}

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

type crossValidationUI struct {
	hessfree.UI

	costFunc          neuralnet.CostFunc
	network           neuralnet.Network
	validationSamples sgd.SampleSet
}

func (c *crossValidationUI) LogCGStart(initQuad, quadLast float64) {
	c.UI.Log("CrossValidation", fmt.Sprintf("cost=%f",
		neuralnet.TotalCost(c.costFunc, c.network, c.validationSamples)))
	c.UI.LogCGStart(initQuad, quadLast)
}
