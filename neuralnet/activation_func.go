package neuralnet

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// Sigmoid is a Layer which applies the
// logistic sigmoid function.
type Sigmoid struct {
	Cache *autofunc.VectorCache
}

func (s *Sigmoid) Apply(r autofunc.Result) autofunc.Result {
	return autofunc.Sigmoid{s.Cache}.Apply(r)
}

func (s *Sigmoid) ApplyR(v autofunc.RVector, r autofunc.RResult) autofunc.RResult {
	return autofunc.Sigmoid{s.Cache}.ApplyR(v, r)
}

func (s *Sigmoid) SetCache(c *autofunc.VectorCache) {
	s.Cache = c
}

func (_ *Sigmoid) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func (_ *Sigmoid) SerializerType() string {
	return serializerTypeSigmoid
}

type ReLU struct {
	Cache *autofunc.VectorCache
}

func (r *ReLU) Apply(in autofunc.Result) autofunc.Result {
	inVec := in.Output()
	vec := r.Cache.Alloc(len(inVec))
	for i, x := range inVec {
		if x > 0 {
			vec[i] = x
		}
	}
	return &reLUResult{
		Cache:     r.Cache,
		OutputVec: vec,
		Input:     in,
	}
}

func (r *ReLU) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	outVec := in.Output()
	outVecR := in.ROutput()
	vec := r.Cache.Alloc(len(outVec))
	vecR := r.Cache.Alloc(len(outVec))
	for i, x := range outVec {
		if x > 0 {
			vec[i] = x
			vecR[i] = outVecR[i]
		}
	}
	return &reLURResult{
		Cache:      r.Cache,
		OutputVec:  vec,
		ROutputVec: vecR,
		Input:      in,
	}
}

func (r *ReLU) SetCache(c *autofunc.VectorCache) {
	r.Cache = c
}

func (r *ReLU) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func (r *ReLU) SerializerType() string {
	return serializerTypeReLU
}

type reLUResult struct {
	Cache     *autofunc.VectorCache
	OutputVec linalg.Vector
	Input     autofunc.Result
}

func (r *reLUResult) Output() linalg.Vector {
	return r.OutputVec
}

func (r *reLUResult) Constant(g autofunc.Gradient) bool {
	return r.Input.Constant(g)
}

func (r *reLUResult) PropagateGradient(upstream linalg.Vector, grad autofunc.Gradient) {
	for i, x := range r.OutputVec {
		if x == 0 {
			upstream[i] = 0
		}
		r.Input.PropagateGradient(upstream, grad)
	}
}

func (r *reLUResult) Release() {
	r.Cache.Free(r.OutputVec)
	r.OutputVec = nil
	r.Input.Release()
}

type reLURResult struct {
	Cache      *autofunc.VectorCache
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector
	Input      autofunc.RResult
}

func (r *reLURResult) Output() linalg.Vector {
	return r.OutputVec
}

func (r *reLURResult) ROutput() linalg.Vector {
	return r.ROutputVec
}

func (r *reLURResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return r.Input.Constant(rg, g)
}

func (r *reLURResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rgrad autofunc.RGradient, grad autofunc.Gradient) {
	for i, x := range r.OutputVec {
		if x == 0 {
			upstream[i] = 0
			upstreamR[i] = 0
		}
		r.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
	}
}

func (r *reLURResult) Release() {
	r.Cache.Free(r.OutputVec)
	r.Cache.Free(r.ROutputVec)
	r.OutputVec = nil
	r.ROutputVec = nil
	r.Input.Release()
}

type HyperbolicTangent struct {
	Cache *autofunc.VectorCache
}

func (h *HyperbolicTangent) Apply(r autofunc.Result) autofunc.Result {
	arith := autofunc.Arithmetic{h.Cache}
	stretched := arith.Scale(autofunc.Sigmoid{h.Cache}.Apply(arith.Scale(r, 2)), 2)
	return arith.AddScaler(stretched, -1)
}

func (h *HyperbolicTangent) ApplyR(v autofunc.RVector, r autofunc.RResult) autofunc.RResult {
	arith := autofunc.Arithmetic{h.Cache}
	stretched := arith.ScaleR(autofunc.Sigmoid{h.Cache}.ApplyR(v, arith.ScaleR(r, 2)), 2)
	return arith.AddScalerR(stretched, -1)
}

func (h *HyperbolicTangent) SetCache(c *autofunc.VectorCache) {
	h.Cache = c
}

func (h *HyperbolicTangent) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func (h *HyperbolicTangent) SerializerType() string {
	return serializerTypeHyperbolicTangent
}
