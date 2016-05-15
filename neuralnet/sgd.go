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

	downstreamGrad := make([]float64, len(n.Output()))
	n.SetDownstreamGradient(downstreamGrad)
	for i := 0; i < s.Epochs; i++ {
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
		for _, j := range order {
			input := s.Inputs[j]
			output := s.Outputs[j]
			n.SetInput(input)
			n.PropagateForward()
			s.CostFunc.Deriv(n, output, downstreamGrad)
			n.PropagateBackward(false)
			s.CostFunc.UpdateInternal(n)
			n.StepGradient(-stepSize)
		}
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
