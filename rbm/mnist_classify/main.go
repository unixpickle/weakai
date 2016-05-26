package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/signal"
	"runtime"

	"github.com/unixpickle/mnist"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rbm"
)

var LayerSizes = []int{28 * 28, 500, 300}

const (
	BoltzmannGibbsSteps = 5
	BoltzmannStepSize   = 1e-2
	BoltzmannEpochs     = 100
	BoltzmannSamples    = 200

	ClassifierStepSize  = 1e-2
	ClassifierMaxEpochs = 10000
	DigitCount          = 10
)

func main() {
	if len(os.Args) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: mnist_classify <classifier_out.json>")
		os.Exit(1)
	}

	outputFile := os.Args[1]

	training := mnist.LoadTrainingDataSet()

	binSamples := binarySamples(training.Samples)
	classifier := pretrainedClassifier(binSamples)

	trainClassifier(classifier, training)
	data := classifier.Serialize()

	if err := ioutil.WriteFile(outputFile, data, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func pretrainedClassifier(d [][]bool) *neuralnet.Network {
	log.Println("Pre-training with DBN...")

	trainer := rbm.Trainer{
		GibbsSteps: BoltzmannGibbsSteps,
		StepSize:   BoltzmannStepSize,
		Epochs:     BoltzmannEpochs,
		BatchSize:  runtime.GOMAXPROCS(0),
	}
	layers := createDBN()
	trainer.TrainDeep(layers, d[:BoltzmannSamples])

	classifier := layers.BuildANN()
	outputLayer := neuralnet.NewDenseLayer(&neuralnet.DenseParams{
		Activation:  neuralnet.Sigmoid{},
		InputCount:  LayerSizes[len(LayerSizes)-1],
		OutputCount: DigitCount,
	})
	outputLayer.Randomize()
	classifier.AddLayer(outputLayer)

	return classifier
}

func trainClassifier(n *neuralnet.Network, d mnist.DataSet) {
	log.Println("Training classifier (ctrl+C to finish)...")

	killChan := make(chan struct{})

	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt)
		<-c
		signal.Stop(c)
		fmt.Println("\nCaught interrupt. Ctrl+C again to terminate.")
		close(killChan)
	}()

	trainer := &neuralnet.SGD{
		CostFunc: neuralnet.MeanSquaredCost{},
		Inputs:   d.IntensityVectors(),
		Outputs:  d.LabelVectors(),
		StepSize: ClassifierStepSize,
		Epochs:   1,
	}

	crossValidation := mnist.LoadTestingDataSet()

	printScore("Initial training", n, d)
	printScore("Initial cross", n, crossValidation)

	for {
		select {
		case <-killChan:
			return
		default:
		}
		trainer.Train(n)
		printScore("Training", n, d)
		printScore("Cross", n, crossValidation)
	}
}

func createDBN() rbm.DBN {
	res := make(rbm.DBN, len(LayerSizes)-1)
	for i := 1; i < len(LayerSizes); i++ {
		res[i-1] = rbm.NewRBM(LayerSizes[i-1], LayerSizes[i])
	}
	return res
}

func binarySamples(tsamp []mnist.Sample) [][]bool {
	samples := make([][]bool, len(tsamp))
	for i, sample := range tsamp {
		samples[i] = make([]bool, len(sample.Intensities))
		for j, x := range sample.Intensities {
			if x > 0.5 {
				samples[i][j] = true
			}
		}
	}
	return samples
}

func printScore(prefix string, n *neuralnet.Network, d mnist.DataSet) {
	classifier := func(v []float64) int {
		n.SetInput(v)
		n.PropagateForward()
		return networkOutput(n)
	}
	correctCount := d.NumCorrect(classifier)
	histogram := d.CorrectnessHistogram(classifier)
	log.Printf("%s: %d/%d - %s", prefix, correctCount, len(d.Samples), histogram)
}

func networkOutput(n *neuralnet.Network) int {
	out := n.Output()
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
