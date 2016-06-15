package rnn

import "github.com/unixpickle/serializer"

const (
	serializerPrefix           = "github.com/unixpickle/weakai/rnn."
	serializerTypeLSTM         = serializerPrefix + "LSTM"
	serializerTypeLSTMGate     = serializerPrefix + "lstmGate"
	serializerTypeStackedBlock = serializerPrefix + "StackedBlock"
)

func init() {
	serializer.RegisterDeserializer(serializerTypeLSTM, DeserializeLSTM)
	serializer.RegisterDeserializer(serializerTypeLSTMGate, deserializeLSTMGate)
	serializer.RegisterDeserializer(serializerTypeStackedBlock, DeserializeStackedBlock)
}
