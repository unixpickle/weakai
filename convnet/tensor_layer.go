package convnet

// tensorLayer is meant to be encapsulated (via
// an anonymous field) by layers with input and
// output 3D tensors.
//
// It implements a subset of the Layer interface,
// but it has no notion of forward or backward
// propagation.
type tensorLayer struct {
	output           *Tensor3
	upstreamGradient *Tensor3

	input              *Tensor3
	downstreamGradient *Tensor3
}

func (t *tensorLayer) Output() []float64 {
	return t.output.Data
}

func (t *tensorLayer) UpstreamGradient() []float64 {
	return t.upstreamGradient.Data
}

func (t *tensorLayer) Input() []float64 {
	if t.input == nil {
		return nil
	}
	return t.input.Data
}

func (t *tensorLayer) SetInput(v []float64) bool {
	if len(v) != len(t.upstreamGradient.Data) {
		return false
	}
	t.input = &Tensor3{
		Width:  t.upstreamGradient.Width,
		Height: t.upstreamGradient.Height,
		Depth:  t.upstreamGradient.Depth,
		Data:   v,
	}
	return true
}

func (t *tensorLayer) DownstreamGradient() []float64 {
	return t.downstreamGradient.Data
}

func (t *tensorLayer) SetDownstreamGradient(v []float64) bool {
	if len(v) != len(t.output.Data) {
		return false
	}
	t.downstreamGradient = &Tensor3{
		Width:  t.output.Width,
		Height: t.output.Height,
		Depth:  t.output.Depth,
		Data:   v,
	}
	return true
}
