package neuralnet

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
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
