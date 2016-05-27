package neuralnet

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"os/signal"

	"github.com/unixpickle/num-analysis/kahan"
)

// SGD trains neural networks using
// gradient descent.
type SGD struct {
	CostFunc CostFunc
	Inputs   [][]float64
	Outputs  [][]float64

	// StepSize indicates how far along each
	// gradient the solver should move.
	StepSize float64

	// StepDecreaseRate specifies how much to
	// decrease the step size after every epoch
	// of gradient descent.
	StepDecreaseRate float64

	// Epochs is the number of rounds of descent
	// the solver should perform before stopping.
	Epochs int

	// BatchSize is the number of input samples
	// per SGD mini-batch.
	// If this is 1 or 0, then pure SGD is used.
	// If this is more than one, then each input
	// in a mini-batch will be handled in its own
	// Goroutine, allowing for potential speedup.
	BatchSize int
}

// Train runs stochastic gradient descent on
// the network.
func (s *SGD) Train(n *Network) {
	s.train(n, false)
}

// TrainInteractive is like Train, but it logs
// its progress and allows the user to send an
// interrupt to stop the training early.
func (s *SGD) TrainInteractive(n *Network) {
	s.train(n, true)
}

func (s *SGD) train(n *Network, interactive bool) {
	killChan := make(chan struct{})

	if interactive {
		go func() {
			c := make(chan os.Signal, 1)
			signal.Notify(c, os.Interrupt)
			<-c
			signal.Stop(c)
			fmt.Println("\nCaught interrupt. Ctrl+C again to terminate.")
			close(killChan)
		}()
	}

	batchSize := s.batchGoroutineCount()

	aliases := make([]Layer, batchSize)
	trainChans := make([]chan int, batchSize)
	doneChan := make(chan struct{}, batchSize)
	aliases[0] = n
	for i := range aliases {
		if i != 0 {
			aliases[i] = n.Alias()
		}
		trainChans[i] = make(chan int)
		go s.trainRoutine(aliases[i], trainChans[i], doneChan)
		defer close(trainChans[i])
	}

	for i := 0; i < s.Epochs || s.Epochs == 0; i++ {
		select {
		case <-killChan:
			log.Println("Finishing due to interrupt")
			return
		default:
		}

		stepSize := s.StepSize - float64(i)*s.StepDecreaseRate
		if stepSize <= 0 {
			break
		}

		if interactive {
			log.Printf("Epoch %d: stepSize = %f; cost = %f", i, stepSize, s.totalCost(n))
		}

		order := rand.Perm(len(s.Inputs))
		for j := 0; j < len(order); j += batchSize {
			size := batchSize
			if j+size > len(order) {
				size = len(order) - j
			}
			for k := 0; k < size; k++ {
				trainChans[k] <- order[j+k]
			}
			for k := 0; k < size; k++ {
				<-doneChan
			}
			for k := 0; k < size; k++ {
				aliases[k].StepGradient(-stepSize)
			}
		}
	}
}

func (s *SGD) trainRoutine(l Layer, inChan <-chan int, doneChan chan<- struct{}) {
	downstreamGrad := make([]float64, len(l.Output()))
	l.SetDownstreamGradient(downstreamGrad)
	for idx := range inChan {
		input := s.Inputs[idx]
		output := s.Outputs[idx]
		l.SetInput(input)
		l.PropagateForward()
		s.CostFunc.Deriv(l, output, downstreamGrad)
		l.PropagateBackward(false)
		s.CostFunc.UpdateInternal(l)
		doneChan <- struct{}{}
	}
}

func (s *SGD) totalCost(n *Network) float64 {
	sum := kahan.NewSummer64()
	for i, in := range s.Inputs {
		n.SetInput(in)
		n.PropagateForward()
		sum.Add(s.CostFunc.Eval(n, s.Outputs[i]))
	}
	return sum.Sum()
}

func (s *SGD) batchGoroutineCount() int {
	if s.BatchSize == 0 {
		return 1
	} else if s.BatchSize > len(s.Inputs) {
		return len(s.Inputs)
	}
	return s.BatchSize
}
