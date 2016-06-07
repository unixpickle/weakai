package neuralnet

import (
	"testing"

	"github.com/unixpickle/serializer"
)

func TestNetworkSerialize(t *testing.T) {
	network := Network{
		&DenseLayer{InputCount: 3, OutputCount: 2},
		Sigmoid{},
	}
	network.Randomize()

	encoded, err := network.Serialize()
	if err != nil {
		t.Fatal(err)
	}
	layerType := network.SerializerType()

	decoded, err := serializer.GetDeserializer(layerType)(encoded)
	if err != nil {
		t.Fatal(err)
	}

	decodedNet, ok := decoded.(Network)
	if !ok {
		t.Fatalf("expected *Network but got %T", decoded)
	}

	if len(network) != len(decodedNet) {
		t.Fatalf("expected %d layers but got %d", len(network), len(decodedNet))
	}

	_, ok = decodedNet[0].(*DenseLayer)
	if !ok {
		t.Fatalf("expected *DenseLayer but got %T", decodedNet[0])
	}

	_, ok = decodedNet[1].(Sigmoid)
	if !ok {
		t.Fatalf("expected Sigmoid but got %T", decodedNet[1])
	}
}
