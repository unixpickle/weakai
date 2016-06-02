package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
	"github.com/unixpickle/weakai/rnn/lstm"
)

const (
	StepSize = 0.1
	Epochs   = 30

	TrainingCount = 100
	TestingCount  = 100
	MaxSeqLen     = 30
	MinSeqLen     = 10

	HiddenSize = 20
)

func main() {
	rand.Seed(time.Now().UnixNano())

	trainer := rnn.RMSProp{
		SGD: rnn.SGD{
			CostFunc: rnn.MeanSquaredCost{},
			StepSize: StepSize,
			Epochs:   Epochs,
		},
	}
	for i := 0; i < TrainingCount; i++ {
		inSeq, outSeq := genEvenOddSeq(rand.Intn(MaxSeqLen-MinSeqLen) + MinSeqLen)
		trainer.InSeqs = append(trainer.InSeqs, inSeq)
		trainer.OutSeqs = append(trainer.OutSeqs, outSeq)
	}
	net := lstm.NewNet(rnn.Sigmoid{}, 2, HiddenSize, 2)
	net.Randomize()
	trainer.Train(net)

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

func runTestSample(ins, outs []linalg.Vector, r *lstm.Net) float64 {
	var correct int
	var total int
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
	r.Reset()

	return float64(correct) / float64(total)
}
