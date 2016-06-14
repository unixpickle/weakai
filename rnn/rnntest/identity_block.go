package rnntest

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/rnn"
)

// An IdentityBlock echos its states and inputs.
type IdentityBlock struct{}

func (_ IdentityBlock) Batch(in *rnn.BlockInput) rnn.BlockOutput {
	return &identityOutput{
		Inputs:   in.Inputs,
		InStates: in.States,
	}
}

func (_ IdentityBlock) BatchR(v autofunc.RVector, in *rnn.BlockRInput) rnn.BlockROutput {
	return &identityROutput{
		Inputs:   in.Inputs,
		InStates: in.States,
	}
}

type identityOutput struct {
	Inputs   []*autofunc.Variable
	InStates []*autofunc.Variable
}

func (i *identityOutput) Outputs() []linalg.Vector {
	res := make([]linalg.Vector, len(i.Inputs))
	for j, x := range i.Inputs {
		res[j] = x.Vector
	}
	return res
}

func (i *identityOutput) States() []linalg.Vector {
	res := make([]linalg.Vector, len(i.InStates))
	for j, x := range i.InStates {
		res[j] = x.Vector
	}
	return res
}

func (i *identityOutput) Gradient(u *rnn.UpstreamGradient, grad autofunc.Gradient) {
	for j, us := range u.Outputs {
		i.Inputs[j].PropagateGradient(us, grad)
	}
	for j, us := range u.States {
		i.InStates[j].PropagateGradient(us, grad)
	}
}

type identityROutput struct {
	Inputs   []*autofunc.RVariable
	InStates []*autofunc.RVariable
}

func (i *identityROutput) Outputs() []linalg.Vector {
	res := make([]linalg.Vector, len(i.Inputs))
	for j, x := range i.Inputs {
		res[j] = x.Variable.Vector
	}
	return res
}

func (i *identityROutput) ROutputs() []linalg.Vector {
	res := make([]linalg.Vector, len(i.Inputs))
	for j, x := range i.Inputs {
		res[j] = x.ROutputVec
	}
	return res
}

func (i *identityROutput) States() []linalg.Vector {
	res := make([]linalg.Vector, len(i.InStates))
	for j, x := range i.InStates {
		res[j] = x.Variable.Vector
	}
	return res
}

func (i *identityROutput) RStates() []linalg.Vector {
	res := make([]linalg.Vector, len(i.InStates))
	for j, x := range i.InStates {
		res[j] = x.ROutputVec
	}
	return res
}

func (i *identityROutput) RGradient(u *rnn.UpstreamRGradient, rgrad autofunc.RGradient,
	grad autofunc.Gradient) {
	for j, us := range u.Outputs {
		i.Inputs[j].PropagateRGradient(us, u.ROutputs[j], rgrad, grad)
	}
	for j, us := range u.States {
		i.InStates[j].PropagateRGradient(us, u.RStates[j], rgrad, grad)
	}
}
