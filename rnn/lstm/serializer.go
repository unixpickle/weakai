package lstm

import "github.com/unixpickle/serializer"

const serializerTypeNet = "github.com/unixpickle/weakai/rnn/lstm.Net"

func init() {
	serializer.RegisterDeserializer(serializerTypeNet, DeserializeNet)
}
