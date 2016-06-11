package neuralnet

import (
	"encoding/json"
	"math"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// DenseLayer is a fully-connected layer of
// linear perceptrons.
// To introduce non-linearities, you may wish
// to follow a DenseLayer with an activation
// function like Sigmoid or ReLU.
type DenseLayer struct {
	InputCount  int
	OutputCount int

	Weights *autofunc.LinTran
	Biases  *autofunc.LinAdd
}

func DeserializeDenseLayer(data []byte) (*DenseLayer, error) {
	var d DenseLayer
	if err := json.Unmarshal(data, &d); err != nil {
		return nil, err
	}
	return &d, nil
}

// Randomize randomizes the weights and biases
// such that the sum of the weights has a mean
// of 0 and a variance of 1.
//
// This will create d.Weights and d.Biases if
// they are nil.
func (d *DenseLayer) Randomize() {
	if d.Biases == nil {
		d.Biases = &autofunc.LinAdd{
			Var: &autofunc.Variable{
				Vector: make(linalg.Vector, d.OutputCount),
			},
		}
	}
	if d.Weights == nil {
		d.Weights = &autofunc.LinTran{
			Rows: d.OutputCount,
			Cols: d.InputCount,
			Data: &autofunc.Variable{
				Vector: make(linalg.Vector, d.OutputCount*d.InputCount),
			},
		}
	}

	sqrt3 := math.Sqrt(3)
	for i := 0; i < d.OutputCount; i++ {
		d.Biases.Var.Vector[i] = sqrt3 * ((rand.Float64() * 2) - 1)
	}

	weightCoeff := math.Sqrt(3.0 / float64(d.InputCount))
	for i := range d.Weights.Data.Vector {
		d.Weights.Data.Vector[i] = weightCoeff * ((rand.Float64() * 2) - 1)
	}
}

// Parameters returns a slice with two variables.
// The first variable contains the weight matrix.
// The second variable contains the bias vector.
func (d *DenseLayer) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{d.Weights.Data, d.Biases.Var}
}

func (d *DenseLayer) Apply(in autofunc.Result) autofunc.Result {
	return d.Biases.Apply(d.Weights.Apply(in))
}

func (d *DenseLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return d.Biases.ApplyR(v, d.Weights.ApplyR(v, in))
}

func (d *DenseLayer) Batch(v autofunc.Result, n int) autofunc.Result {
	biasBatcher := &autofunc.FuncBatcher{F: d.Biases}
	return biasBatcher.Batch(d.Weights.Batch(v, n), n)
}

func (d *DenseLayer) BatchR(rv autofunc.RVector, v autofunc.RResult, n int) autofunc.RResult {
	biasBatcher := &autofunc.RFuncBatcher{F: d.Biases}
	return biasBatcher.BatchR(rv, d.Weights.BatchR(rv, v, n), n)
}

func (d *DenseLayer) Serialize() ([]byte, error) {
	return json.Marshal(d)
}

func (d *DenseLayer) SerializerType() string {
	return serializerTypeDenseLayer
}
