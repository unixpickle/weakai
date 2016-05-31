package rnn

import "github.com/unixpickle/serializer"

const (
	serializerTypePrefix  = "github.com/unixpickle/weakai/rnn."
	serializerTypeSigmoid = serializerTypePrefix + "Sigmoid"
	serializerTypeReLU    = serializerTypePrefix + "ReLU"
	serializerTypeTanh    = serializerTypePrefix + "Tanh"
	serializerTypeDeepRNN = serializerTypePrefix + "DeepRNN"
)

func init() {
	serializer.RegisterDeserializer(serializerTypeSigmoid, fixedDeserializer(Sigmoid{}))
	serializer.RegisterDeserializer(serializerTypeReLU, fixedDeserializer(ReLU{}))
	serializer.RegisterDeserializer(serializerTypeTanh, fixedDeserializer(Tanh{}))
	serializer.RegisterDeserializer(serializerTypeDeepRNN, DeserializeDeepRNN)
}

func fixedDeserializer(obj serializer.Serializer) serializer.Deserializer {
	return func(d []byte) (serializer.Serializer, error) {
		return obj, nil
	}
}
