package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/seqtoseq"
)

const (
	StepSize     = 1e-3
	StateSize    = 40
	HiddenSize   = 10
	BatchSize    = 5
	TrainingSize = 3000
	TestingSize  = 100
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
		Forward:  &rnn.BlockSeqFunc{B: rnn.NewGRU(2, StateSize)},
		Backward: &rnn.BlockSeqFunc{B: rnn.NewGRU(2, StateSize)},
		Output:   &rnn.NetworkSeqFunc{Network: outNet},
	}
	var samples []seqtoseq.Sample
	var sampleSet sgd.SliceSampleSet
	for i := 0; i < TrainingSize; i++ {
		samples = append(samples, generateSequence())
		sampleSet = append(sampleSet, samples[i])
	}

	g := &sgd.RMSProp{
		Gradienter: &seqtoseq.Gradienter{
			SeqFunc:  bd,
			Learner:  bd,
			CostFunc: neuralnet.DotCost{},
		},
	}

	var i int
	sgd.SGDInteractive(g, sampleSet, StepSize, BatchSize, func() bool {
		fmt.Printf("%d epochs: cost=%f\n", i, totalCost(bd, sampleSet))
		i++
		return true
	})

	var testingCorrect, testingTotal int
	for j := 0; j < TestingSize; j++ {
		sample := generateSequence()
		inRes := seqfunc.ConstResult([][]linalg.Vector{sample.Inputs})
		output := bd.ApplySeqs(inRes).OutputSeqs()[0]
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

func sgdOnSequences(f *rnn.Bidirectional, s []seqtoseq.Sample) {
	gradient := autofunc.NewGradient(f.Parameters())
	for _, x := range s {
		inRes := seqfunc.ConstResult([][]linalg.Vector{x.Inputs})
		output := f.ApplySeqs(inRes)
		upstreamGrad := make([]linalg.Vector, len(x.Outputs))
		for i, o := range x.Outputs {
			upstreamGrad[i] = o.Copy().Scale(-1)
		}
		output.PropagateGradient([][]linalg.Vector{upstreamGrad}, gradient)
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

func generateSequence() seqtoseq.Sample {
	var seq seqtoseq.Sample
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

func totalCost(f *rnn.Bidirectional, s sgd.SampleSet) float64 {
	return seqtoseq.TotalCostSeqFunc(f, 10, s, neuralnet.DotCost{})
}
