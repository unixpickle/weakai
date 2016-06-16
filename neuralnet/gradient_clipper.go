package neuralnet

import (
	"math"

	"github.com/unixpickle/autofunc"
)

// GradientClipper is a Gradienter which scales down
// gradients so that the L2 norm of the gradient is
// less than a certain value.
type GradientClipper struct {
	Gradienter Gradienter
	Threshold  float64
}

func (c *GradientClipper) Gradient(s SampleSet) autofunc.Gradient {
	res := c.Gradienter.Gradient(s)
	var magnitude float64
	for _, vec := range res {
		magnitude += vec.Dot(vec)
	}
	magnitude = math.Sqrt(magnitude)
	if magnitude > c.Threshold {
		res.Scale(c.Threshold / magnitude)
	}
	return res
}
