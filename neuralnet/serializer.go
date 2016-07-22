package neuralnet

import "github.com/unixpickle/serializer"

const (
	serializerTypePrefix            = "github.com/unixpickle/weakai/neuralnet."
	serializerTypeHyperbolicTangent = serializerTypePrefix + "HyperbolicTangent"
	serializerTypeSigmoid           = serializerTypePrefix + "Sigmoid"
	serializerTypeBorderLayer       = serializerTypePrefix + "BorderLayer"
	serializerTypeUnstackLayer      = serializerTypePrefix + "UnstackLayer"
	serializerTypeConvLayer         = serializerTypePrefix + "ConvLayer"
	serializerTypeDenseLayer        = serializerTypePrefix + "DenseLayer"
	serializerTypeMaxPoolingLayer   = serializerTypePrefix + "MaxPoolingLayer"
	serializerTypeSoftmaxLayer      = serializerTypePrefix + "SoftmaxLayer"
	serializerTypeLogSoftmaxLayer   = serializerTypePrefix + "LogSoftmaxLayer"
	serializerTypeNetwork           = serializerTypePrefix + "Network"
	serializerTypeReLU              = serializerTypePrefix + "ReLU"
	serializerTypeRescaleLayer      = serializerTypePrefix + "RescaleLayer"
	serializerTypeDropoutLayer      = serializerTypePrefix + "DropoutLayer"
	serializerTypeVecRescaleLayer   = serializerTypePrefix + "VecRescaleLayer"
	serializerTypeGaussNoiseLayer   = serializerTypePrefix + "GaussNoiseLayer"
)

func init() {
	serializer.RegisterDeserializer(serializerTypeSigmoid,
		func(d []byte) (serializer.Serializer, error) {
			return &Sigmoid{}, nil
		})
	serializer.RegisterDeserializer(serializerTypeReLU,
		func(d []byte) (serializer.Serializer, error) {
			return &ReLU{}, nil
		})
	serializer.RegisterDeserializer(serializerTypeHyperbolicTangent,
		func(d []byte) (serializer.Serializer, error) {
			return &HyperbolicTangent{}, nil
		})
	serializer.RegisterTypedDeserializer(serializerTypeConvLayer,
		DeserializeConvLayer)
	serializer.RegisterTypedDeserializer(serializerTypeDenseLayer,
		DeserializeDenseLayer)
	serializer.RegisterTypedDeserializer(serializerTypeNetwork,
		DeserializeNetwork)
	serializer.RegisterTypedDeserializer(serializerTypeBorderLayer,
		DeserializeBorderLayer)
	serializer.RegisterTypedDeserializer(serializerTypeSoftmaxLayer,
		DeserializeSoftmaxLayer)
	serializer.RegisterTypedDeserializer(serializerTypeLogSoftmaxLayer,
		DeserializeLogSoftmaxLayer)
	serializer.RegisterTypedDeserializer(serializerTypeMaxPoolingLayer,
		DeserializeMaxPoolingLayer)
	serializer.RegisterTypedDeserializer(serializerTypeUnstackLayer,
		DeserializeUnstackLayer)
	serializer.RegisterTypedDeserializer(serializerTypeRescaleLayer,
		DeserializeRescaleLayer)
	serializer.RegisterTypedDeserializer(serializerTypeDropoutLayer,
		DeserializeDropoutLayer)
	serializer.RegisterTypedDeserializer(serializerTypeVecRescaleLayer,
		DeserializeVecRescaleLayer)
	serializer.RegisterTypedDeserializer(serializerTypeGaussNoiseLayer,
		DeserializeGaussNoiseLayer)
}
