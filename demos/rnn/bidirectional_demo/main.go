package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	StepSize     = 1e-3
	StateSize    = 40
	HiddenSize   = 10
	BatchSize    = 100
	TrainingSize = 3000
	TestingSize  = 100
	Epochs       = 100
)

func main() {
	rand.Seed(time.Now().UnixNano())

	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  StateSize * 2,
			OutputCount: 10,
		},
		&neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  10,
			OutputCount: 2,
		},
		&neuralnet.LogSoftmaxLayer{},
	}
	outNet.Randomize()
	bd := &rnn.Bidirectional{
		Forward:  &rnn.RNNSeqFunc{Block: rnn.NewGRU(2, StateSize)},
		Backward: &rnn.RNNSeqFunc{Block: rnn.NewGRU(2, StateSize)},
		Output:   &rnn.NetworkSeqFunc{Network: outNet},
	}
	var samples []rnn.Sequence
	for i := 0; i < TrainingSize; i++ {
		samples = append(samples, generateSequence())
	}
	for i := 0; i < Epochs; i++ {
		fmt.Printf("%d epochs: cost=%f\n", i, totalCost(bd, samples))
		for k := 0; k+BatchSize < len(samples); k += BatchSize {
			sgdOnSequences(bd, samples[k:k+BatchSize])
		}
	}

	var testingCorrect, testingTotal int
	for j := 0; j < TestingSize; j++ {
		sample := generateSequence()
		output := bd.BatchSeqs(inputResultsForSeq(sample)).OutputSeqs()[0]
		for i, expected := range sample.Outputs {
			actual := output[i]
			if math.Abs(expected[0]-math.Exp(actual[0])) < 0.1 {
				testingCorrect++
			}
			testingTotal++
		}
	}

	fmt.Printf("Got %d/%d (%.2f%%)\n", testingCorrect, testingTotal,
		100*float64(testingCorrect)/float64(testingTotal))
}

func sgdOnSequences(f *rnn.Bidirectional, s []rnn.Sequence) {
	gradient := autofunc.NewGradient(f.Parameters())
	for _, x := range s {
		output := f.BatchSeqs(inputResultsForSeq(x))
		upstreamGrad := make([]linalg.Vector, len(x.Outputs))
		for i, o := range x.Outputs {
			upstreamGrad[i] = o.Copy().Scale(-1)
		}
		output.Gradient([][]linalg.Vector{upstreamGrad}, gradient)
	}
	for _, vec := range gradient {
		for i, x := range vec {
			if x > 0 {
				vec[i] = 1
			} else {
				vec[i] = -1
			}
		}
	}
	gradient.AddToVars(-StepSize)
}

func generateSequence() rnn.Sequence {
	var seq rnn.Sequence
	for i := -10; i < rand.Intn(20); i++ {
		if rand.Intn(2) == 0 {
			seq.Inputs = append(seq.Inputs, linalg.Vector{1, 0})
		} else {
			seq.Inputs = append(seq.Inputs, linalg.Vector{0, 1})
		}
	}
	for i := range seq.Inputs {
		var leftSet, rightSet bool
		if i > 0 {
			leftSet = seq.Inputs[i-1][0] == 1
		}
		if i+1 < len(seq.Inputs) {
			rightSet = seq.Inputs[i+1][0] == 1
		}
		if (leftSet && !rightSet) || (!leftSet && rightSet) {
			seq.Outputs = append(seq.Outputs, linalg.Vector{1, 0})
		} else {
			seq.Outputs = append(seq.Outputs, linalg.Vector{0, 1})
		}
	}
	return seq
}

func totalCost(f *rnn.Bidirectional, s []rnn.Sequence) float64 {
	var sum float64
	for _, sample := range s {
		output := f.BatchSeqs(inputResultsForSeq(sample))
		for i, o := range sample.Outputs {
			sum += o.Dot(output.OutputSeqs()[0][i])
		}
	}
	return -sum
}

func inputResultsForSeq(s rnn.Sequence) [][]autofunc.Result {
	var res []autofunc.Result
	for _, x := range s.Inputs {
		res = append(res, &autofunc.Variable{Vector: x})
	}
	return [][]autofunc.Result{res}
}
