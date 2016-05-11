package convnet

import "fmt"

// A Layer represents any kind of "layer"
// that a deep neural network may have.
type Layer interface {
	PropagateForward()
	PropagateBackward()
}

// A Network is nothing more than a stack
// of layers, one on top of each other.
type Network struct {
	Layers []Layer
}

// NewNetwork generates a network by creating
// and connecting a bunch of layers.
//
// The paramList argument is a list of structs
// of type *DenseParams, *ConvParams, or
// *MaxPoolingParams, corresponding to the
// configuration of that layer.
//
// The first layer is considered input layer,
// and the final one is considered the
// output layer.
//
// It is invalid to connect the output of a dense
// layer to the input of any other kind of layer,
// so convolutional and max pooling layers should
// precede dense layers.
//
// If the structure is invalid, this returns an error.
func NewNetwork(paramList []interface{}) (*Network, error) {
	n := &Network{Layers: make([]Layer, len(paramList))}
	for i, param := range paramList {
		switch param := param.(type) {
		case *DenseParams:
			l := NewDenseLayer(param)
			if i == 0 {
				l.Input = make([]float64, param.InputCount)
			} else {
				l.Input = layerOutputVector(n.Layers[i-1])
			}
			n.Layers[i] = l
		case *ConvParams:
			l := NewConvLayer(param)
			if i == 0 {
				l.Input = NewTensor3(param.InputWidth, param.InputHeight, param.InputDepth)
			} else {
				tensor, err := layerOutputTensor3(n.Layers[i-1])
				if err != nil {
					return nil, err
				} else if tensor.Width != param.InputWidth || tensor.Height != param.InputHeight ||
					tensor.Depth != param.InputDepth {
					return nil, fmt.Errorf("incorrect input dimensions for layer %d", i)
				}
				l.Input = tensor
			}
			n.Layers[i] = l
		case *MaxPoolingParams:
			l := NewMaxPoolingLayer(param)
			if i == 0 {
				l.Input = NewTensor3(param.InputWidth, param.InputHeight, param.InputDepth)
			} else {
				tensor, err := layerOutputTensor3(n.Layers[i-1])
				if err != nil {
					return nil, err
				} else if tensor.Width != param.InputWidth || tensor.Height != param.InputHeight ||
					tensor.Depth != param.InputDepth {
					return nil, fmt.Errorf("incorrect input dimensions for layer %d", i)
				}
				l.Input = tensor
			}
			n.Layers[i] = l
		default:
			return nil, fmt.Errorf("unknown type for params[%d]: %T", i, param)
		}
	}
	n.connectGradients()
	return n, nil
}

func (n *Network) PropagateForward() {
	for _, l := range n.Layers {
		l.PropagateForward()
	}
}

func (n *Network) PropagateBackward() {
	for _, l := range n.Layers {
		l.PropagateBackward()
	}
}

func (n *Network) connectGradients() {
	for i, layer := range n.Layers[:len(n.Layers)-1] {
		nextLayer := n.Layers[i+1]
		switch layer := layer.(type) {
		case *DenseLayer:
			layer.DownstreamGradient = layerUpstreamGradientVector(nextLayer)
		case *MaxPoolingLayer:
			ot := layer.Output
			layer.DownstreamGradient = layerUpstreamGradientTensor3(nextLayer, ot.Width,
				ot.Height, ot.Depth)
		case *ConvLayer:
			ot := layer.Output
			layer.DownstreamGradient = layerUpstreamGradientTensor3(nextLayer, ot.Width,
				ot.Height, ot.Depth)
		}
	}

	lastLayer := n.Layers[len(n.Layers)-1]
	switch lastLayer := lastLayer.(type) {
	case *DenseLayer:
		lastLayer.DownstreamGradient = make([]float64, len(lastLayer.Output))
	case *MaxPoolingLayer:
		ot := lastLayer.Output
		lastLayer.DownstreamGradient = NewTensor3(ot.Width, ot.Height, ot.Depth)
	case *ConvLayer:
		ot := lastLayer.Output
		lastLayer.DownstreamGradient = NewTensor3(ot.Width, ot.Height, ot.Depth)
	}
}

func layerOutputVector(l Layer) []float64 {
	switch l := l.(type) {
	case *DenseLayer:
		return l.Output
	case *MaxPoolingLayer:
		return l.Output.Data
	case *ConvLayer:
		return l.Output.Data
	default:
		panic(fmt.Sprintf("unknown layer type: %T", l))
	}
}

func layerOutputTensor3(l Layer) (*Tensor3, error) {
	switch l := l.(type) {
	case *DenseLayer:
		return nil, fmt.Errorf("layer type does not output a Tensor3: %T", l)
	case *MaxPoolingLayer:
		return l.Output, nil
	case *ConvLayer:
		return l.Output, nil
	default:
		panic(fmt.Sprintf("unknown layer type: %T", l))
	}
}

func layerUpstreamGradientVector(l Layer) []float64 {
	switch l := l.(type) {
	case *DenseLayer:
		return l.UpstreamGradient
	default:
		panic(fmt.Sprintf("type %T does not have upstream gradient vector", l))
	}
}

func layerUpstreamGradientTensor3(l Layer, width, height, depth int) *Tensor3 {
	switch l := l.(type) {
	case *DenseLayer:
		return &Tensor3{
			Width:  width,
			Height: height,
			Depth:  depth,
			Data:   l.UpstreamGradient,
		}
	case *ConvLayer:
		return l.UpstreamGradient
	case *MaxPoolingLayer:
		return l.UpstreamGradient
	default:
		panic(fmt.Sprintf("unknown layer type: %T", l))
	}
}
