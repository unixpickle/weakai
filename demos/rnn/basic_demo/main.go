package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

const (
	StepSize  = 0.05
	Epochs    = 60
	BatchSize = 10

	TrainingCount = 100
	TestingCount  = 100
	MaxSeqLen     = 30
	MinSeqLen     = 10

	HiddenSize = 20
)

func main() {
	rand.Seed(time.Now().UnixNano())

	sampleSet := neuralnet.SliceSampleSet{}
	for i := 0; i < TrainingCount; i++ {
		inSeq, outSeq := genEvenOddSeq(rand.Intn(MaxSeqLen-MinSeqLen) + MinSeqLen)
		sampleSet = append(sampleSet, rnn.Sequence{
			Inputs:  inSeq,
			Outputs: outSeq,
		})
	}

	outNet := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  HiddenSize,
			OutputCount: 2,
		},
	}
	outNet.Randomize()
	outBlock := rnn.NewNetworkBlock(outNet, 0)
	lstm := rnn.NewLSTM(2, HiddenSize)
	net := rnn.StackedBlock{lstm, outBlock}

	gradienter := &neuralnet.RMSProp{
		Gradienter: &rnn.BPTT{
			Learner:  net,
			CostFunc: neuralnet.SigmoidCECost{},
			MaxLanes: 1,
		},
	}

	neuralnet.SGD(gradienter, sampleSet, StepSize, Epochs, BatchSize)

	outNet = append(outNet, neuralnet.Sigmoid{})

	var scoreSum float64
	var scoreTotal float64
	for i := 0; i < TestingCount; i++ {
		size := rand.Intn(MaxSeqLen-MinSeqLen) + MinSeqLen
		ins, outs := genEvenOddSeq(size)
		score := runTestSample(ins, outs, net)
		scoreSum += score
		scoreTotal += 1
	}

	fmt.Println("Testing success rate:", scoreSum/scoreTotal)
}

func genEvenOddSeq(size int) (ins, outs []linalg.Vector) {
	var odds [2]bool
	for i := 0; i < size; i++ {
		sampleIdx := rand.Intn(2)
		odds[sampleIdx] = !odds[sampleIdx]
		if sampleIdx == 0 {
			ins = append(ins, linalg.Vector{1, 0})
		} else {
			ins = append(ins, linalg.Vector{0, 1})
		}
		out := linalg.Vector{0, 0}
		for j, o := range odds {
			if o {
				out[j] = 1
			}
		}
		outs = append(outs, out)
	}
	return
}

func runTestSample(ins, outs []linalg.Vector, b rnn.Block) float64 {
	var correct int
	var total int
	r := &rnn.Runner{Block: b}
	for i, in := range ins {
		out := r.StepTime(in)
		if out[0] < 0.5 == (outs[i][0] < 0.5) {
			correct++
		}
		if out[1] < 0.5 == (outs[i][1] < 0.5) {
			correct++
		}
		total += 2
	}

	return float64(correct) / float64(total)
}
