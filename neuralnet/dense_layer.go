package neuralnet

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

var denseLayerByteOrder = binary.LittleEndian

const denseLayerDataVersion byte = '2'

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
	// Backwards-compatible JSON-based layer data.
	if len(data) == 0 || data[0] != denseLayerDataVersion {
		var d DenseLayer
		if err := json.Unmarshal(data, &d); err != nil {
			return nil, err
		}
		return &d, nil
	}

	reader := bytes.NewBuffer(data[1:])
	var inCount int64
	var outCount int64
	if err := binary.Read(reader, denseLayerByteOrder, &inCount); err != nil {
		return nil, err
	}
	if err := binary.Read(reader, denseLayerByteOrder, &outCount); err != nil {
		return nil, err
	}

	res := &DenseLayer{
		InputCount:  int(inCount),
		OutputCount: int(outCount),
	}

	weightCount := res.InputCount * res.OutputCount
	biasCount := res.OutputCount
	dataSize := 8 * (weightCount + biasCount)
	if reader.Len() != dataSize {
		return nil, fmt.Errorf("expected %d DenseLayer bytes but have %d",
			dataSize, reader.Len())
	}

	res.Weights = &autofunc.LinTran{
		Data: &autofunc.Variable{Vector: make(linalg.Vector, weightCount)},
		Rows: res.OutputCount,
		Cols: res.InputCount,
	}
	for i := 0; i < weightCount; i++ {
		paramPtr := &res.Weights.Data.Vector[i]
		if err := binary.Read(reader, denseLayerByteOrder, paramPtr); err != nil {
			return nil, err
		}
	}

	res.Biases = &autofunc.LinAdd{
		Var: &autofunc.Variable{Vector: make(linalg.Vector, biasCount)},
	}
	for i := 0; i < biasCount; i++ {
		paramPtr := &res.Biases.Var.Vector[i]
		if err := binary.Read(reader, denseLayerByteOrder, paramPtr); err != nil {
			return nil, err
		}
	}

	return res, nil
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
	if d.Weights == nil || d.Biases == nil {
		panic(uninitPanicMessage)
	}
	return []*autofunc.Variable{d.Weights.Data, d.Biases.Var}
}

func (d *DenseLayer) Apply(in autofunc.Result) autofunc.Result {
	if d.Weights == nil || d.Biases == nil {
		panic(uninitPanicMessage)
	}
	return d.Biases.Apply(d.Weights.Apply(in))
}

func (d *DenseLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	if d.Weights == nil || d.Biases == nil {
		panic(uninitPanicMessage)
	}
	return d.Biases.ApplyR(v, d.Weights.ApplyR(v, in))
}

func (d *DenseLayer) Batch(v autofunc.Result, n int) autofunc.Result {
	if d.Weights == nil || d.Biases == nil {
		panic(uninitPanicMessage)
	}
	biasBatcher := &autofunc.FuncBatcher{F: d.Biases}
	return biasBatcher.Batch(d.Weights.Batch(v, n), n)
}

func (d *DenseLayer) BatchR(rv autofunc.RVector, v autofunc.RResult, n int) autofunc.RResult {
	if d.Weights == nil || d.Biases == nil {
		panic(uninitPanicMessage)
	}
	biasBatcher := &autofunc.RFuncBatcher{F: d.Biases}
	return biasBatcher.BatchR(rv, d.Weights.BatchR(rv, v, n), n)
}

func (d *DenseLayer) Serialize() ([]byte, error) {
	if d.Weights == nil || d.Biases == nil {
		panic(uninitPanicMessage)
	}
	weightCount := d.InputCount * d.OutputCount
	biasCount := d.OutputCount
	b := make([]byte, 0, 17+8*(weightCount+biasCount))
	resBuf := bytes.NewBuffer(b)

	resBuf.WriteByte(denseLayerDataVersion)
	binary.Write(resBuf, denseLayerByteOrder, uint64(d.InputCount))
	binary.Write(resBuf, denseLayerByteOrder, uint64(d.OutputCount))
	for _, w := range d.Weights.Data.Vector {
		binary.Write(resBuf, denseLayerByteOrder, w)
	}
	for _, w := range d.Biases.Var.Vector {
		binary.Write(resBuf, denseLayerByteOrder, w)
	}

	return resBuf.Bytes(), nil
}

func (d *DenseLayer) SerializerType() string {
	return serializerTypeDenseLayer
}
