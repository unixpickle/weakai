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
func SGD(g Gradienter, samples SampleSet, stepSize float64, epochs, batchSize int) {
	s := samples.Copy()
	for i := 0; i < epochs; i++ {
		s.Shuffle()
		for j := 0; j < len(s); j += batchSize {
			count := batchSize
			if count > len(s)-j {
				count = len(s) - j
			}
			subset := s[j : j+count]
			grad := g.Gradient(subset)
			grad.AddToVars(-stepSize)
		}
	}
}

// SGDInteractive is like SGD, but it calls a
// function before every epoch and stops when
// said function returns false, or when the
// user sends a kill signal.
func SGDInteractive(g Gradienter, s SampleSet, stepSize float64, batchSize int, sf func() bool) {
	var killed uint32

	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	defer func() {
		select {
		case <-c:
		default:
			signal.Stop(c)
			close(c)
		}
	}()

	go func() {
		_, ok := <-c
		if !ok {
			return
		}
		signal.Stop(c)
		close(c)
		atomic.StoreUint32(&killed, 1)
		fmt.Println("\nCaught interrupt. Ctrl+C again to terminate.")
	}()

	for atomic.LoadUint32(&killed) == 0 {
		if sf != nil {
			if !sf() {
				return
			}
		}
		if atomic.LoadUint32(&killed) != 0 {
			return
		}
		SGD(g, s, stepSize, 1, batchSize)
	}
}
