package neuralnet

import (
	"reflect"

	"github.com/unixpickle/serializer"
)

const (
	serializerTypePrefix            = "github.com/unixpickle/weakai/neuralnet."
	serializerTypeHyperbolicTangent = serializerTypePrefix + "HyperbolicTangent"
	serializerTypeSigmoid           = serializerTypePrefix + "Sigmoid"
	serializerTypeBorderLayer       = serializerTypePrefix + "BorderLayer"
	serializerTypeConvGrowLayer     = serializerTypePrefix + "ConvGrowLayer"
	serializerTypeConvLayer         = serializerTypePrefix + "ConvLayer"
	serializerTypeDenseLayer        = serializerTypePrefix + "DenseLayer"
	serializerTypeMaxPoolingLayer   = serializerTypePrefix + "MaxPoolingLayer"
	serializerTypeSoftmaxLayer      = serializerTypePrefix + "SoftmaxLayer"
	serializerTypeNetwork           = serializerTypePrefix + "Network"
)

func init() {
	serializer.RegisterDeserializer(serializerTypeSigmoid,
		func(d []byte) (serializer.Serializer, error) {
			return Sigmoid{}, nil
		})
	serializer.RegisterDeserializer(serializerTypeHyperbolicTangent,
		func(d []byte) (serializer.Serializer, error) {
			return HyperbolicTangent{}, nil
		})
	serializer.RegisterDeserializer(serializerTypeConvLayer,
		convertDeserializer(DeserializeConvLayer))
	serializer.RegisterDeserializer(serializerTypeDenseLayer,
		convertDeserializer(DeserializeDenseLayer))
	serializer.RegisterDeserializer(serializerTypeMaxPoolingLayer,
		convertDeserializer(DeserializeMaxPoolingLayer))
	serializer.RegisterDeserializer(serializerTypeNetwork,
		convertDeserializer(DeserializeNetwork))
	serializer.RegisterDeserializer(serializerTypeBorderLayer,
		convertDeserializer(DeserializeBorderLayer))
	serializer.RegisterDeserializer(serializerTypeConvGrowLayer,
		convertDeserializer(DeserializeConvGrowLayer))
	serializer.RegisterDeserializer(serializerTypeSoftmaxLayer,
		convertDeserializer(DeserializeSoftmaxLayer))
}

func convertDeserializer(f interface{}) serializer.Deserializer {
	val := reflect.ValueOf(f)
	return func(d []byte) (serializer.Serializer, error) {
		res := val.Call([]reflect.Value{reflect.ValueOf(d)})
		if res[1].IsNil() {
			return res[0].Interface().(serializer.Serializer), nil
		} else {
			return nil, res[1].Interface().(error)
		}
	}
}
