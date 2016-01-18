package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

func main() {
	rand.Seed(time.Now().UnixNano())
	firstBitTest()
}

// firstBitTest builds a neural network which outputs 0 for
// inputs starting with a 1 and 0 for inputs starting with a 1.
func firstBitTest() {
	var maxError float64
	var encounteredError float64

	for test := 0; test < 50; test++ {
		network := NewNetwork(4, 1, 0, 1, SigmoidFunction{})
		for i := 0; i < 1000; i++ {
			input := make([]float64, 4)
			for j := range input {
				input[j] = float64(rand.Intn(2))
			}
			desired := 1 - input[0]
			network.SetInput(input)
			network.Adjust([]float64{desired}, 10)
		}
		for i := 0; i < 1000; i++ {
			maxError += 1
			input := make([]float64, 4)
			for j := range input {
				input[j] = float64(rand.Intn(2))
			}
			desired := 1 - input[0]
			network.SetInput(input)
			output := network.Evaluate()[0]
			encounteredError += math.Abs(desired - output)
		}
	}
	fmt.Println("Average errors:", encounteredError/maxError)
}
