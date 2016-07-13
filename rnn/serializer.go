package rnn

import "github.com/unixpickle/serializer"

const (
	serializerPrefix             = "github.com/unixpickle/weakai/rnn."
	serializerTypeLSTM           = serializerPrefix + "LSTM"
	serializerTypeLSTMGate       = serializerPrefix + "lstmGate"
	serializerTypeStackedBlock   = serializerPrefix + "StackedBlock"
	serializerTypeNetworkBlock   = serializerPrefix + "NetworkBlock"
	serializerTypeGRU            = serializerPrefix + "GRU"
	serializerTypeRNNSeqFunc     = serializerPrefix + "RNNSeqFunc"
	serializerTypeNetworkSeqFunc = serializerPrefix + "NetworkSeqFunc"
)

func init() {
	serializer.RegisterTypedDeserializer(serializerTypeLSTM, DeserializeLSTM)
	serializer.RegisterTypedDeserializer(serializerTypeLSTMGate, deserializeLSTMGate)
	serializer.RegisterTypedDeserializer(serializerTypeStackedBlock, DeserializeStackedBlock)
	serializer.RegisterTypedDeserializer(serializerTypeNetworkBlock, DeserializeNetworkBlock)
	serializer.RegisterTypedDeserializer(serializerTypeGRU, DeserializeGRU)
	serializer.RegisterTypedDeserializer(serializerTypeRNNSeqFunc, DeserializeRNNSeqFunc)
	serializer.RegisterTypedDeserializer(serializerTypeNetworkSeqFunc, DeserializeNetworkSeqFunc)
}
