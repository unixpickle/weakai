package neuralnet

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/unixpickle/num-analysis/kahan"
)

var networkEncodingEndian = binary.LittleEndian

// A Network is a feed-forward neural network
// composed of a series of layers, each one
// feeding into the next one.
//
// A Network itself acts as a Layer, with
// its input feeding into the first layer
// and its output coming out of the last one.
type Network struct {
	Layers []Layer
}

// NewNetwork generates a network by creating
// and connecting a bunch of layers using their
// respective prototypes.
//
// If layer dimensions do not match, this returns
// an error describing the issue.
func NewNetwork(prototypes []LayerPrototype) (*Network, error) {
	n := &Network{Layers: make([]Layer, 0, len(prototypes))}
	for _, proto := range prototypes {
		layer := proto.Make()
		if err := n.AddLayer(layer); err != nil {
			return nil, err
		}
	}
	return n, nil
}

func DeserializeNetwork(data []byte) (*Network, error) {
	buffer := bytes.NewBuffer(data)

	var count uint32
	if err := binary.Read(buffer, networkEncodingEndian, &count); err != nil {
		return nil, err
	}

	res := &Network{make([]Layer, 0, int(count))}

	for i := 0; i < int(count); i++ {
		var typeLen uint32
		if err := binary.Read(buffer, networkEncodingEndian, &typeLen); err != nil {
			return nil, err
		}
		typeData := make([]byte, int(typeLen))
		n, err := buffer.Read(typeData)
		if err != nil {
			return nil, err
		} else if n < len(typeData) {
			return nil, errors.New("buffer underflow")
		}

		var dataLen uint64
		if err := binary.Read(buffer, networkEncodingEndian, &dataLen); err != nil {
			return nil, err
		}
		layerData := make([]byte, int(dataLen))
		n, err = buffer.Read(layerData)
		if err != nil {
			return nil, err
		} else if n < len(layerData) {
			return nil, errors.New("buffer underflow")
		}

		typeStr := string(typeData)
		decoder, ok := Deserializers[typeStr]
		if !ok {
			return nil, fmt.Errorf("unknown serializer type: %s", typeStr)
		}
		layerObj, err := decoder(layerData)
		if err != nil {
			return nil, fmt.Errorf("failed to decode layer %d: %s", i, err.Error())
		} else if _, ok := layerObj.(Layer); !ok {
			return nil, fmt.Errorf("layer %d is not Layer, it's %T", i, layerObj)
		}

		if err := res.AddLayer(layerObj.(Layer)); err != nil {
			return nil, err
		}
	}

	return res, nil
}

// AddLayer adds a layer to the tail of this network.
// This fails if the layer's dimensionality does not
// match the previous layer's output dimensions.
func (n *Network) AddLayer(l Layer) error {
	idx := len(n.Layers)
	n.Layers = append(n.Layers, l)
	if idx != 0 {
		ok := n.Layers[idx].SetInput(n.Layers[idx-1].Output())
		if !ok {
			return fmt.Errorf("layer %d cannot feed into layer %d", idx-1, idx)
		}
		ok = n.Layers[idx-1].SetDownstreamGradient(n.Layers[idx].UpstreamGradient())
		if !ok {
			return fmt.Errorf("layer %d cannot feed back into layer %d", idx, idx-1)
		}
	}
	return nil
}

func (n *Network) Randomize() {
	for _, layer := range n.Layers {
		layer.Randomize()
	}
}

func (n *Network) PropagateForward() {
	for _, l := range n.Layers {
		l.PropagateForward()
	}
}

func (n *Network) PropagateBackward(upstream bool) {
	for i := len(n.Layers) - 1; i >= 0; i-- {
		if i != 0 || upstream {
			n.Layers[i].PropagateBackward(true)
		} else {
			n.Layers[i].PropagateBackward(false)
		}
	}
}

func (n *Network) Output() []float64 {
	return n.Layers[len(n.Layers)-1].Output()
}

func (n *Network) UpstreamGradient() []float64 {
	return n.Layers[0].UpstreamGradient()
}

func (n *Network) Input() []float64 {
	return n.Layers[0].Input()
}

func (n *Network) SetInput(v []float64) bool {
	return n.Layers[0].SetInput(v)
}

func (n *Network) DownstreamGradient() []float64 {
	return n.Layers[len(n.Layers)-1].DownstreamGradient()
}

func (n *Network) SetDownstreamGradient(v []float64) bool {
	return n.Layers[len(n.Layers)-1].SetDownstreamGradient(v)
}

func (n *Network) GradientMagSquared() float64 {
	sum := kahan.NewSummer64()
	for _, l := range n.Layers {
		sum.Add(l.GradientMagSquared())
	}
	return sum.Sum()
}

func (n *Network) StepGradient(f float64) {
	for _, l := range n.Layers {
		l.StepGradient(f)
	}
}

func (n *Network) Serialize() []byte {
	var buf bytes.Buffer

	binary.Write(&buf, networkEncodingEndian, uint32(len(n.Layers)))

	for _, l := range n.Layers {
		encoded := l.Serialize()
		typeName := []byte(l.SerializerType())

		binary.Write(&buf, networkEncodingEndian, uint32(len(typeName)))
		buf.Write(typeName)

		binary.Write(&buf, networkEncodingEndian, uint64(len(encoded)))
		buf.Write(encoded)
	}

	return buf.Bytes()
}

func (n *Network) SerializerType() string {
	return "network"
}
