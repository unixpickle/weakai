package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/boosting"
)

func main() {
	rand.Seed(time.Now().UnixNano())

	data := RandomSampleList(80)
	classes := RandomClassVector(data.Len())

	grad := boosting.Gradient{
		Loss:    boosting.SquareLoss{},
		Desired: classes,
		List:    data,
		Pool:    NewStumpPool(data),
	}
	var i int
	for {
		loss := grad.Step()
		errs := errorCount(&grad.Sum, data, classes)
		fmt.Println("Epoch:", i, "loss:", loss, "errors:", errs)
		if errs == 0 {
			break
		}
		i++
	}
}

func errorCount(c boosting.Classifier, l SampleList, classes linalg.Vector) int {
	var count int
	for i, a := range c.Classify(l) {
		if (a < 0) != (classes[i] < 0) {
			count++
		}
	}
	return count
}
