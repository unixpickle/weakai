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
}

func NewGradient(inSize, hiddenSize, outSize int) *Gradient {
	return &Gradient{
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
	}
}

func (r *Gradient) Add(r1 *Gradient) {
	r.OutWeights.Add(r1.OutWeights)
	r.OutBiases.Add(r1.OutBiases)
	r.InWeights.Add(r1.InWeights)
	r.InGate.Add(r1.InGate)
	r.RemGate.Add(r1.RemGate)
	r.OutGate.Add(r1.OutGate)
	r.InBiases.Add(r1.InBiases)
	r.InGateBiases.Add(r1.InGateBiases)
	r.RemGateBiases.Add(r1.RemGateBiases)
	r.OutGateBiases.Add(r1.OutGateBiases)
}
