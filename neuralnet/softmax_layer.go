package neuralnet

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type SoftmaxLayer autofunc.Softmax

func DeserializeSoftmaxLayer(d []byte) (*SoftmaxLayer, error) {
	var res SoftmaxLayer
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

func (s *SoftmaxLayer) Apply(in autofunc.Result) autofunc.Result {
	soft := (*autofunc.Softmax)(s)
	return soft.Apply(in)
}

func (s *SoftmaxLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	soft := (*autofunc.Softmax)(s)
	return soft.ApplyR(v, in)
}

func (s *SoftmaxLayer) Serialize() ([]byte, error) {
	return json.Marshal(s)
}

func (s *SoftmaxLayer) SerializerType() string {
	return serializerTypeSoftmaxLayer
}

// LogSoftmaxLayer is a layer which returns the log
// of the softmax function (with temperature = 1).
type LogSoftmaxLayer struct{}

func (s *LogSoftmaxLayer) Apply(in autofunc.Result) autofunc.Result {
	return autofunc.Pool(in, func(in autofunc.Result) autofunc.Result {
		// Compute the log of the sum of the exponents by
		// factoring out the largest exponent so that all
		// the exponentials fit nicely inside floats.
		maxIdx := maxVecIdx(in.Output())
		maxValue := autofunc.Slice(in, maxIdx, maxIdx+1)
		exponents := autofunc.AddFirst(in, autofunc.Scale(maxValue, -1))
		expSum := autofunc.SumAll(autofunc.Exp{}.Apply(exponents))
		expLog := autofunc.Log{}.Apply(expSum)
		denomLog := autofunc.Add(expLog, maxValue)
		return autofunc.AddFirst(in, autofunc.Scale(denomLog, -1))
	})
}

func (s *LogSoftmaxLayer) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return autofunc.PoolR(in, func(in autofunc.RResult) autofunc.RResult {
		// See comment in Apply() for details on how this works.
		maxIdx := maxVecIdx(in.Output())
		maxValue := autofunc.SliceR(in, maxIdx, maxIdx+1)
		exponents := autofunc.AddFirstR(in, autofunc.ScaleR(maxValue, -1))
		expSum := autofunc.SumAllR(autofunc.Exp{}.ApplyR(v, exponents))
		expLog := autofunc.Log{}.ApplyR(v, expSum)
		denomLog := autofunc.AddR(expLog, maxValue)
		return autofunc.AddFirstR(in, autofunc.ScaleR(denomLog, -1))
	})
}

func maxVecIdx(v linalg.Vector) int {
	var maxVal float64
	var maxIdx int
	for i, x := range v {
		if i == 0 {
			maxVal = x
			maxIdx = i
		} else if x > maxVal {
			maxVal = x
			maxIdx = i
		}
	}
	return maxIdx
}
