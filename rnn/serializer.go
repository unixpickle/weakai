package rnn

import "github.com/unixpickle/serializer"

const (
	serializerPrefix             = "github.com/unixpickle/weakai/rnn."
	serializerTypeLSTM           = serializerPrefix + "LSTM"
	serializerTypeLSTMGate       = serializerPrefix + "lstmGate"
	serializerTypeStackedBlock   = serializerPrefix + "StackedBlock"
	serializerTypeNetworkBlock   = serializerPrefix + "NetworkBlock"
	serializerTypeGRU            = serializerPrefix + "GRU"
	serializerTypeBlockSeqFunc   = serializerPrefix + "BlockSeqFunc"
	serializerTypeNetworkSeqFunc = serializerPrefix + "NetworkSeqFunc"
	serializerTypeBidirectional  = serializerPrefix + "Bidirectional"
)

func init() {
	serializer.RegisterTypedDeserializer(serializerTypeLSTM, DeserializeLSTM)
	serializer.RegisterTypedDeserializer(serializerTypeLSTMGate, deserializeLSTMGate)
	serializer.RegisterTypedDeserializer(serializerTypeStackedBlock, DeserializeStackedBlock)
	serializer.RegisterTypedDeserializer(serializerTypeNetworkBlock, DeserializeNetworkBlock)
	serializer.RegisterTypedDeserializer(serializerTypeGRU, DeserializeGRU)
	serializer.RegisterTypedDeserializer(serializerTypeBlockSeqFunc, DeserializeBlockSeqFunc)
	serializer.RegisterTypedDeserializer(serializerTypeNetworkSeqFunc, DeserializeNetworkSeqFunc)
	serializer.RegisterTypedDeserializer(serializerTypeBidirectional, DeserializeBidirectional)
}
