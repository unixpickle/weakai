package softmax

import "github.com/unixpickle/serializer"

const serializerTypeSoftmax = "github.com/unixpickle/weakai/rnn/softmax.Softmax"

func init() {
	serializer.RegisterDeserializer(serializerTypeSoftmax, DeserializeSoftmax)
}
