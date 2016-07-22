package neuralnet

import (
	"encoding/json"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A GaussNoiseLayer adds random gaussian noise to
// its input vectors with the given standard
// deviation.
// This is similar to DropoutLayer, except that it
// modifies the inputs instead of dropping them.
type GaussNoiseLayer struct {
	// Stddev is the standard devation of the noise
	// added to the inputs.
	Stddev float64

	// Training is true if noise should be applied.
	Training bool
}

func DeserializeGaussNoiseLayer(d []byte) (*GaussNoiseLayer, error) {
	var res GaussNoiseLayer
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

func (g *GaussNoiseLayer) Apply(in autofunc.Result) autofunc.Result {
	if g.Training {
		return autofunc.Add(in, g.noise(len(in.Output())))
	} else {
		return in
	}
}

func (g *GaussNoiseLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	if g.Training {
		return autofunc.AddR(in, g.noiseR(len(in.Output())))
	} else {
		return in
	}
}

func (g *GaussNoiseLayer) SerializerType() string {
	return serializerTypeGaussNoiseLayer
}

func (g *GaussNoiseLayer) Serialize() ([]byte, error) {
	return json.Marshal(g)
}

func (g *GaussNoiseLayer) noise(size int) autofunc.Result {
	vec := make(linalg.Vector, size)
	for i := range vec {
		vec[i] = rand.NormFloat64() * g.Stddev
	}
	return &autofunc.Variable{Vector: vec}
}

func (g *GaussNoiseLayer) noiseR(size int) autofunc.RResult {
	vec := make(linalg.Vector, size)
	for i := range vec {
		vec[i] = rand.NormFloat64() * g.Stddev
	}
	return &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: vec},
		ROutputVec: make(linalg.Vector, size),
	}
}
