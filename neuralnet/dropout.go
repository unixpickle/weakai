package neuralnet

import (
	"encoding/json"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// DropoutLayer implements dropout regularization,
// where random inputs are "dropped" each time the
// layer is evaluated.
//
// A DropoutLayer can either be in training mode,
// where inputs are dropped stochastically, or in
// usage mode, where inputs are scaled to output
// their expected values.
//
// Unlike normal autofunc.RFuncs, a DropoutLayer in
// training mode may return different values each
// time it is evaluated.
// As a result, it will most likely fail traditional
// autofunc tests which assume consistent functions.
type DropoutLayer struct {
	// KeepProbability is the probability that an
	// individual input is not dropped at each
	// function evaluation.
	KeepProbability float64

	// Training is true if inputs should be dropped
	// stochastically rather than averaged.
	Training bool
}

func DeserializeDropoutLayer(d []byte) (*DropoutLayer, error) {
	var res DropoutLayer
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

func (d *DropoutLayer) Apply(in autofunc.Result) autofunc.Result {
	if d.Training {
		return autofunc.Mul(in, d.dropoutMask(len(in.Output())))
	} else {
		return autofunc.Scale(in, d.KeepProbability)
	}
}

func (d *DropoutLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	if d.Training {
		mask := d.dropoutMask(len(in.Output()))
		maskVar := autofunc.NewRVariable(mask, v)
		return autofunc.MulR(in, maskVar)
	} else {
		return autofunc.ScaleR(in, d.KeepProbability)
	}
}

func (d *DropoutLayer) Serialize() ([]byte, error) {
	return json.Marshal(d)
}

func (d *DropoutLayer) SerializerType() string {
	return serializerTypeDropoutLayer
}

func (d *DropoutLayer) dropoutMask(inLen int) *autofunc.Variable {
	resVec := make(linalg.Vector, inLen)
	for i := range resVec {
		if rand.Float64() > d.KeepProbability {
			resVec[i] = 0
		} else {
			resVec[i] = 1
		}
	}
	return &autofunc.Variable{resVec}
}
