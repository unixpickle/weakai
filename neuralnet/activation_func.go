package neuralnet

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// Sigmoid is a Layer which applies the
// logistic sigmoid function.
type Sigmoid struct{}

func (_ Sigmoid) Apply(r autofunc.Result) autofunc.Result {
	return autofunc.Sigmoid{}.Apply(r)
}

func (_ Sigmoid) ApplyR(v autofunc.RVector, r autofunc.RResult) autofunc.RResult {
	return autofunc.Sigmoid{}.ApplyR(v, r)
}

func (_ Sigmoid) Batch(inputs autofunc.Result, n int) autofunc.Result {
	return Sigmoid{}.Apply(inputs)
}

func (_ Sigmoid) BatchR(v autofunc.RVector, inputs autofunc.RResult, n int) autofunc.RResult {
	return Sigmoid{}.ApplyR(v, inputs)
}

func (_ Sigmoid) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func (_ Sigmoid) SerializerType() string {
	return serializerTypeSigmoid
}

type ReLU struct{}

func (_ ReLU) Apply(r autofunc.Result) autofunc.Result {
	inVec := r.Output()
	vec := make(linalg.Vector, len(inVec))
	for i, x := range inVec {
		if x > 0 {
			vec[i] = x
		}
	}
	return &reLUResult{
		OutputVec: vec,
		Input:     r,
	}
}

func (_ ReLU) ApplyR(v autofunc.RVector, r autofunc.RResult) autofunc.RResult {
	outVec := r.Output()
	outVecR := r.ROutput()
	vec := make(linalg.Vector, len(outVec))
	vecR := make(linalg.Vector, len(outVec))
	for i, x := range outVec {
		if x > 0 {
			vec[i] = x
			vecR[i] = outVecR[i]
		}
	}
	return &reLURResult{
		OutputVec:  vec,
		ROutputVec: vecR,
		Input:      r,
	}
}

func (_ ReLU) Batch(inputs autofunc.Result, n int) autofunc.Result {
	return ReLU{}.Apply(inputs)
}

func (_ ReLU) BatchR(v autofunc.RVector, inputs autofunc.RResult, n int) autofunc.RResult {
	return ReLU{}.ApplyR(v, inputs)
}

func (_ ReLU) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func (_ ReLU) SerializerType() string {
	return serializerTypeReLU
}

type reLUResult struct {
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
	if r.Input.Constant(grad) {
		return
	}
	for i, x := range r.OutputVec {
		if x == 0 {
			upstream[i] = 0
		}
	}
	r.Input.PropagateGradient(upstream, grad)
}

type reLURResult struct {
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
	if r.Input.Constant(rgrad, grad) {
		return
	}
	for i, x := range r.OutputVec {
		if x == 0 {
			upstream[i] = 0
			upstreamR[i] = 0
		}
	}
	r.Input.PropagateRGradient(upstream, upstreamR, rgrad, grad)
}

type HyperbolicTangent struct{}

func (_ HyperbolicTangent) Apply(r autofunc.Result) autofunc.Result {
	stretched := autofunc.Scale(autofunc.Sigmoid{}.Apply(autofunc.Scale(r, 2)), 2)
	return autofunc.AddScaler(stretched, -1)
}

func (_ HyperbolicTangent) ApplyR(v autofunc.RVector, r autofunc.RResult) autofunc.RResult {
	stretched := autofunc.ScaleR(autofunc.Sigmoid{}.ApplyR(v, autofunc.ScaleR(r, 2)), 2)
	return autofunc.AddScalerR(stretched, -1)
}

func (_ HyperbolicTangent) Batch(inputs autofunc.Result, n int) autofunc.Result {
	return HyperbolicTangent{}.Apply(inputs)
}

func (_ HyperbolicTangent) BatchR(v autofunc.RVector, inputs autofunc.RResult,
	n int) autofunc.RResult {
	return HyperbolicTangent{}.ApplyR(v, inputs)
}

func (_ HyperbolicTangent) Serialize() ([]byte, error) {
	return []byte{}, nil
}

func (_ HyperbolicTangent) SerializerType() string {
	return serializerTypeHyperbolicTangent
}

type Sin struct {
	autofunc.Sin
}

func (_ Sin) Batch(inputs autofunc.Result, n int) autofunc.Result {
	return Sin{}.Apply(inputs)
}

func (_ Sin) BatchR(v autofunc.RVector, inputs autofunc.RResult, n int) autofunc.RResult {
	return Sin{}.ApplyR(v, inputs)
}

func (_ Sin) SerializerType() string {
	return serializerTypeSin
}

func (_ Sin) Serialize() ([]byte, error) {
	return []byte{}, nil
}
