package main

import (
	"fmt"
	"math"
	"math/rand"
	"time"
)

const GridSize = 4

func main() {
	rand.Seed(time.Now().UnixNano())
	firstBitTest()
	horizontalLineTest()
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
	fmt.Println("firstBitTest() error rate:", encounteredError/maxError, "want <0.5")
}

// horizontalLineTest builds a neural network which rejects
// bitmaps that have horizontal lines.
func horizontalLineTest() {
	var maxError float64
	var encounteredError float64

	for test := 0; test < 10; test++ {
		network := NewNetwork(GridSize*GridSize, 1, 0, 10, SigmoidFunction{})
		for i := 0; i < 500000; i++ {
			input := make([]float64, GridSize*GridSize)
			for j := range input {
				input[j] = float64(rand.Intn(2))
			}
			desired := 1.0
			if bitmapHasHorizontal(input) {
				desired = 0.0
			}
			network.SetInput(input)
			network.Adjust([]float64{desired}, 1)
		}
		for i := 0; i < 1000; i++ {
			maxError += 1
			input := make([]float64, GridSize*GridSize)
			for j := range input {
				input[j] = float64(rand.Intn(2))
			}
			desired := 1.0
			if bitmapHasHorizontal(input) {
				desired = 0.0
			}
			network.SetInput(input)
			output := network.Evaluate()[0]
			encounteredError += math.Abs(desired - output)
		}
	}
	probRejected := 1.0 - math.Pow((math.Pow(2, GridSize)-1)/math.Pow(2, GridSize), GridSize)
	fmt.Println("horizontalLineTest() error rate:", encounteredError/maxError, "want <",
		probRejected)
}

func bitmapHasHorizontal(bitmap []float64) bool {
	for y := 0; y < GridSize; y++ {
		broken := false
		for x := 0; x < GridSize; x++ {
			if bitmap[y*GridSize+x] < 0.5 {
				broken = true
				break
			}
		}
		if !broken {
			return true
		}
	}
	return false
}
