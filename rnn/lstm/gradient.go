package lstm

import "github.com/unixpickle/num-analysis/linalg"

type Gradient struct {
	OutWeights *linalg.Matrix
	OutBiases  linalg.Vector

	InWeights *linalg.Matrix
	InGate    *linalg.Matrix
	RemGate   *linalg.Matrix
	OutGate   *linalg.Matrix

	InBiases      linalg.Vector
	InGateBiases  linalg.Vector
	RemGateBiases linalg.Vector
	OutGateBiases linalg.Vector

	InputGrads []linalg.Vector
}

func NewGradient(inSize, hiddenSize, outSize, time int) *Gradient {
	res := &Gradient{
		OutWeights: linalg.NewMatrix(outSize, hiddenSize+inSize),
		OutBiases:  make(linalg.Vector, outSize),

		InWeights: linalg.NewMatrix(hiddenSize, hiddenSize+inSize),
		InGate:    linalg.NewMatrix(hiddenSize, hiddenSize+inSize),
		RemGate:   linalg.NewMatrix(hiddenSize, hiddenSize+inSize),
		OutGate:   linalg.NewMatrix(hiddenSize, hiddenSize+inSize),

		InBiases:      make(linalg.Vector, hiddenSize),
		InGateBiases:  make(linalg.Vector, hiddenSize),
		RemGateBiases: make(linalg.Vector, hiddenSize),
		OutGateBiases: make(linalg.Vector, hiddenSize),

		InputGrads: make([]linalg.Vector, time),
	}
	for i := range res.InputGrads {
		res.InputGrads[i] = make(linalg.Vector, inSize)
	}
	return res
}

func (r *Gradient) Inputs() []linalg.Vector {
	return r.InputGrads
}

func (r *Gradient) Params() []linalg.Vector {
	return []linalg.Vector{
		linalg.Vector(r.OutWeights.Data),
		linalg.Vector(r.InWeights.Data),
		linalg.Vector(r.InGate.Data),
		linalg.Vector(r.RemGate.Data),
		linalg.Vector(r.OutGate.Data),
		r.OutBiases,
		r.InBiases,
		r.InGateBiases,
		r.RemGateBiases,
		r.OutGateBiases,
	}
}
