package rnn

import (
	"math/rand"
	"runtime"

	"github.com/unixpickle/num-analysis/linalg"
)

// SGD is a set of training parameters for
// mini-batch stochastic gradient descent.
type SGD struct {
	// InSeqs stores the input sequences to
	// train the RNN on.
	InSeqs [][]linalg.Vector

	// OutSeqs stores the output sequences
	// to train the RNN on.
	OutSeqs [][]linalg.Vector

	// CostFunc is the function which SGD
	// should aim to minimize.
	CostFunc CostFunc

	// BatchSize is the number of sequences to
	// use for each step of gradient descent.
	// If this is 0, a value of 1 is used.
	//
	// For each batch, one gradient is computed
	// by summing the gradients for each sequence
	// in the batch.
	BatchSize int

	// StepSize is a value by which each gradient
	// is scaled before descent.
	StepSize float64

	// Epochs is the number of times the trainer
	// should go through all of the training
	// sequences before terminating.
	Epochs int

	// Normalizer is an optional routine which
	// performs a transformation on each mini-batch
	// gradient before a step of descent.
	// This could be used to implement things like
	// RMSProp or weight momentum.
	//
	// This may be nil to use the straightup gradient
	// at each step of descent.
	Normalizer func(g Gradient)
}

func (s *SGD) Train(r RNN) {
	if s.BatchSize == 1 || s.BatchSize == 0 || runtime.GOMAXPROCS(0) < 2 {
		s.TrainSynchronously(r)
	}

	inChan := make(chan int, s.BatchSize)
	outChan := make(chan Gradient, s.BatchSize)

	defer close(inChan)

	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		go func() {
			net := r.Alias()
			for sampleIdx := range inChan {
				seq := s.InSeqs[sampleIdx]
				outSeq := s.OutSeqs[sampleIdx]
				var costPartials []linalg.Vector
				for k, input := range seq {
					actualOut := net.StepTime(input)
					costGrad := make(linalg.Vector, len(actualOut))
					s.CostFunc.Gradient(actualOut, outSeq[k], costGrad)
					costPartials = append(costPartials, costGrad)
				}
				outChan <- net.CostGradient(costPartials)
			}
		}()
	}

	for i := 0; i < s.Epochs; i++ {
		perm := rand.Perm(len(s.InSeqs))
		for j := 0; j < len(perm); j += s.BatchSize {
			batchSize := s.BatchSize
			if j+batchSize > len(perm) {
				batchSize = len(perm) - j
			}
			for k := 0; k < batchSize; k++ {
				inChan <- perm[k+j]
			}
			var gradSum Gradient
			for k := 0; k < batchSize; k++ {
				if k == 0 {
					gradSum = <-outChan
				} else {
					AddGradients(gradSum, <-outChan)
				}
			}
			s.stepGrad(r, gradSum)
		}
	}
}

// TrainSynchronously is like Train, but it will
// not use separate Goroutines to concurrently
// compute gradients for mini-batches.
// This may be faster than Train() when GOMAXPROCS
// is 1, or when BatchSize is 1.
func (s *SGD) TrainSynchronously(r RNN) {
	batchSize := s.BatchSize
	if batchSize == 0 {
		batchSize = 1
	}

	for i := 0; i < s.Epochs; i++ {
		perm := rand.Perm(len(s.InSeqs))
		var batchSum Gradient
		for samplesDone, j := range perm {
			seq := s.InSeqs[j]
			outSeq := s.OutSeqs[j]
			var costPartials []linalg.Vector
			for k, input := range seq {
				actualOut := r.StepTime(input)
				costGrad := make(linalg.Vector, len(actualOut))
				s.CostFunc.Gradient(actualOut, outSeq[k], costGrad)
				costPartials = append(costPartials, costGrad)
			}
			grad := r.CostGradient(costPartials)
			if batchSum != nil {
				AddGradients(batchSum, grad)
			} else {
				batchSum = grad
			}
			if (samplesDone+1)%batchSize == 0 {
				s.stepGrad(r, batchSum)
				batchSum = nil
			}
		}
		if batchSum != nil {
			s.stepGrad(r, batchSum)
		}
	}
}

func (s *SGD) stepGrad(r RNN, grad Gradient) {
	if s.Normalizer != nil {
		s.Normalizer(grad)
	}
	ScaleGradient(grad, -s.StepSize)
	r.StepGradient(grad)
}
