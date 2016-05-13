package neuralnet

import "testing"

func TestNetworkSerialize(t *testing.T) {
	network, _ := NewNetwork([]LayerPrototype{
		&ConvParams{
			Activation:   Sigmoid{},
			FilterCount:  4,
			FilterWidth:  2,
			FilterHeight: 2,
			Stride:       1,
			InputWidth:   4,
			InputHeight:  4,
			InputDepth:   1,
		},
		&DenseParams{
			Activation:  Sigmoid{},
			InputCount:  3 * 3 * 4,
			OutputCount: 4,
		},
		&DenseParams{
			Activation:  Sigmoid{},
			InputCount:  4,
			OutputCount: 1,
		},
	})

	encoded := network.Serialize()
	layerType := network.SerializerType()

	decoded, err := Deserializers[layerType](encoded)
	if err != nil {
		t.Fatal(err)
	}

	network, ok := decoded.(*Network)
	if !ok {
		t.Fatalf("expected *Network but got %T", decoded)
	}

	if len(network.Layers) != 3 {
		t.Fatalf("expected 3 layers but got %d", len(network.Layers))
	}

	_, ok = network.Layers[0].(*ConvLayer)
	if !ok {
		t.Fatalf("expected *ConvLayer but got %T", network.Layers[0])
	}

	for i := 1; i < 3; i++ {
		_, ok = network.Layers[i].(*DenseLayer)
		if !ok {
			t.Fatalf("expected *DenseLayer but got %T at %d", network.Layers[0], i)
		}
	}
}
