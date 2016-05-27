package neuralnet

import "encoding/json"

// BorderParams stores parameters used
// to configure a BorderLayer.
type BorderParams struct {
	InputWidth  int
	InputHeight int
	InputDepth  int

	LeftBorder   int
	RightBorder  int
	TopBorder    int
	BottomBorder int
}

// Make creates a new BorderLayer using
// the params defined by p.
// This is equivalent to NewBorderLayer(p).
func (p *BorderParams) Make() Layer {
	return NewBorderLayer(p)
}

// A BorderLayer takes a 3D tensor as
// input and adds horizontal and vertical
// padding to its edges.
type BorderLayer struct {
	tensorLayer

	leftBorder int
	topBorder  int
}

func NewBorderLayer(params *BorderParams) *BorderLayer {
	outWidth := params.InputWidth + params.LeftBorder + params.RightBorder
	outHeight := params.InputHeight + params.TopBorder + params.BottomBorder
	return &BorderLayer{
		tensorLayer: tensorLayer{
			output: NewTensor3(outWidth, outHeight, params.InputDepth),
			upstreamGradient: NewTensor3(params.InputWidth, params.InputHeight,
				params.InputDepth),
		},
		leftBorder: params.LeftBorder,
		topBorder:  params.TopBorder,
	}
}

func DeserializeBorderLayer(data []byte) (*BorderLayer, error) {
	var s serializedBorderLayer
	if err := json.Unmarshal(data, &s); err != nil {
		return nil, err
	}
	return &BorderLayer{
		tensorLayer: tensorLayer{
			output:           NewTensor3(s.OutWidth, s.OutHeight, s.Depth),
			upstreamGradient: NewTensor3(s.InWidth, s.InHeight, s.Depth),
		},
		leftBorder: s.LeftBorder,
		topBorder:  s.TopBorder,
	}, nil
}

func (b *BorderLayer) Randomize() {
}

func (b *BorderLayer) PropagateForward() {
	for y := 0; y < b.input.Height; y++ {
		for x := 0; x < b.input.Width; x++ {
			for z := 0; z < b.input.Depth; z++ {
				num := b.input.Get(x, y, z)
				b.output.Set(x+b.leftBorder, y+b.topBorder, z, num)
			}
		}
	}
}

func (b *BorderLayer) PropagateBackward(upstream bool) {
	for y := 0; y < b.input.Height; y++ {
		for x := 0; x < b.input.Width; x++ {
			for z := 0; z < b.input.Depth; z++ {
				num := b.downstreamGradient.Get(x+b.leftBorder, y+b.topBorder, z)
				b.upstreamGradient.Set(x, y, z, num)
			}
		}
	}
}

func (b *BorderLayer) GradientMagSquared() float64 {
	return 0
}

func (b *BorderLayer) StepGradient(f float64) {
}

func (b *BorderLayer) Alias() Layer {
	return &BorderLayer{
		tensorLayer: tensorLayer{
			output: NewTensor3(b.output.Width, b.output.Height, b.output.Depth),
			upstreamGradient: NewTensor3(b.upstreamGradient.Width, b.upstreamGradient.Height,
				b.upstreamGradient.Depth),
		},
		leftBorder: b.leftBorder,
		topBorder:  b.topBorder,
	}
}

func (b *BorderLayer) Serialize() []byte {
	s := serializedBorderLayer{
		Depth:      b.output.Depth,
		InWidth:    b.upstreamGradient.Width,
		InHeight:   b.upstreamGradient.Height,
		OutWidth:   b.output.Width,
		OutHeight:  b.output.Height,
		LeftBorder: b.leftBorder,
		TopBorder:  b.topBorder,
	}
	enc, _ := json.Marshal(&s)
	return enc
}

func (b *BorderLayer) SerializerType() string {
	return "borderlayer"
}

type serializedBorderLayer struct {
	Depth int

	InWidth   int
	InHeight  int
	OutWidth  int
	OutHeight int

	LeftBorder int
	TopBorder  int
}
