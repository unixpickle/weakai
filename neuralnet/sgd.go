package neuralnet

import (
	"fmt"
	"os"
	"os/signal"
	"sync/atomic"
)

// SGD trains the Learner of a Batcher using
// stochastic gradient descent with the provided
// inputs and corresponding expected outputs.
func SGD(b *Batcher, s *SampleSet, stepSize float64, epochs, batchSize int) {
	for i := 0; i < epochs; i++ {
		for j := 0; j < len(s.Inputs); j += batchSize {
			count := batchSize
			if count > len(s.Inputs)-j {
				count = len(s.Inputs) - j
			}
			subset := s.Subset(j, j+count)
			grad := b.BatchGradient(subset)
			grad.AddToVars(-stepSize)
		}
	}
}

// SGDInteractive is like SGD, but it calls a
// function before every epoch and stops when
// the user sends a kill signal.
func SGDInteractive(b *Batcher, s *SampleSet, stepSize float64, batchSize int, sf func()) {
	var killed uint32
	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt)
		<-c
		signal.Stop(c)
		atomic.StoreUint32(&killed, 1)
		fmt.Println("\nCaught interrupt. Ctrl+C again to terminate.")
	}()

	for atomic.LoadUint32(&killed) == 0 {
		if sf != nil {
			sf()
		}
		if atomic.LoadUint32(&killed) != 0 {
			break
		}
		SGD(b, s, stepSize, 1, batchSize)
	}
}
