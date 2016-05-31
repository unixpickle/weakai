package softmax

import (
	"strconv"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/weakai/neuralnet"
	"github.com/unixpickle/weakai/rnn"
)

type Softmax struct {
	Size int

	softmaxLayers []*neuralnet.SoftmaxLayer
}

func NewSoftmax(size int) *Softmax {
	return &Softmax{Size: size}
}

func DeserializeSoftmax(d []byte) (serializer.Serializer, error) {
	num, err := strconv.Atoi(string(d))
	if err != nil {
		return nil, err
	}
	return NewSoftmax(num), nil
}

func (s *Softmax) Randomize() {
}

func (s *Softmax) StepTime(input linalg.Vector) linalg.Vector {
	layer := neuralnet.NewSoftmaxLayer(&neuralnet.SoftmaxParams{Size: s.Size})
	layer.SetInput(input)
	layer.PropagateForward()
	s.softmaxLayers = append(s.softmaxLayers, layer)
	return layer.Output()
}

func (s *Softmax) CostGradient(outGrads []linalg.Vector) rnn.Gradient {
	var res Gradient
	for i, layer := range s.softmaxLayers {
		layer.SetDownstreamGradient(outGrads[i])
		layer.PropagateBackward(true)
		res = append(res, layer.UpstreamGradient())
	}
	return res
}

func (s *Softmax) Reset() {
	s.softmaxLayers = nil
}

func (s *Softmax) StepGradient(g rnn.Gradient) {
	s.Reset()
}

func (s *Softmax) Alias() rnn.RNN {
	return NewSoftmax(s.Size)
}

func (s *Softmax) Serialize() ([]byte, error) {
	return []byte(strconv.Itoa(s.Size)), nil
}

func (s *Softmax) SerializerType() string {
	return serializerTypeSoftmax
}
