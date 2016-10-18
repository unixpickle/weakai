package rnn

import (
	"errors"
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
)

func init() {
	var b Bidirectional
	serializer.RegisterTypedDeserializer(b.SerializerType(), DeserializeBidirectional)
}

// Bidirectional facilitates architectures like the
// bidirectional RNN described in
// http://arxiv.org/pdf/1303.5778.pdf.
//
// For example, you could implement a bidirectional LSTM
// by using two BlockSeqFuncs (forward and backward) with
// two LSTM blocks, and a NetworkSeqFunc for the output.
//
// If the input sequence is of length N, Output will
// be given an input of length N which contains packed
// time steps.
// Each time step fed into Output is packed with the
// forward outputs followed by the backward outputs.
type Bidirectional struct {
	Forward  seqfunc.RFunc
	Backward seqfunc.RFunc
	Output   seqfunc.RFunc
}

// DeserializeBidirectional deserializes a previously
// serialized Bidirectional instance.
func DeserializeBidirectional(d []byte) (*Bidirectional, error) {
	slice, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}
	if len(slice) != 3 {
		return nil, errors.New("invalid Bidirectional slice length")
	}
	s1, ok1 := slice[0].(seqfunc.RFunc)
	s2, ok2 := slice[1].(seqfunc.RFunc)
	s3, ok3 := slice[2].(seqfunc.RFunc)
	if !ok1 || !ok2 || !ok3 {
		return nil, errors.New("invalid Bidirectional slice types")
	}
	return &Bidirectional{s1, s2, s3}, nil
}

// ApplySeqs applies the bidirectional RNN to an input.
func (b *Bidirectional) ApplySeqs(in seqfunc.Result) seqfunc.Result {
	return seqfunc.Pool(in, func(in seqfunc.Result) seqfunc.Result {
		forward := b.Forward.ApplySeqs(in)
		backward := seqfunc.Reverse(b.Backward.ApplySeqs(in))
		return b.Output.ApplySeqs(seqfunc.ConcatInner(forward, backward))
	})
}

// ApplySeqsR applies the bidirectional RNN to an input.
func (b *Bidirectional) ApplySeqsR(rv autofunc.RVector, in seqfunc.RResult) seqfunc.RResult {
	return seqfunc.PoolR(in, func(in seqfunc.RResult) seqfunc.RResult {
		forward := b.Forward.ApplySeqsR(rv, in)
		backward := seqfunc.ReverseR(b.Backward.ApplySeqsR(rv, in))
		return b.Output.ApplySeqsR(rv, seqfunc.ConcatInnerR(forward, backward))
	})
}

// Parameters combines the parameters of all three
// internal seqfunc.RFuncs, ignoring the ones that don't
// implement sgd.Learner.
func (b *Bidirectional) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	for _, x := range []seqfunc.RFunc{b.Forward, b.Backward, b.Output} {
		if l, ok := x.(sgd.Learner); ok {
			res = append(res, l.Parameters()...)
		}
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// Bidirectional instances with the serializer package.
func (b *Bidirectional) SerializerType() string {
	return "github.com/unixpickle/weakai/rnn.Bidirectional"
}

// Serialize attempts to serialize b.
// This fails if any of the seqfunc.RFuncs are not
// serializer.Serializers.
func (b *Bidirectional) Serialize() ([]byte, error) {
	var slice []serializer.Serializer
	for _, x := range []seqfunc.RFunc{b.Forward, b.Backward, b.Output} {
		s, ok := x.(serializer.Serializer)
		if !ok {
			return nil, fmt.Errorf("type cannot be serialized: %T", x)
		}
		slice = append(slice, s)
	}
	return serializer.SerializeSlice(slice)
}
