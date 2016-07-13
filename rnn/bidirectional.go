package rnn

import (
	"errors"
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/sgd"
)

// Bidirectional facilitates architectures like the
// bidirectional RNN described in
// http://arxiv.org/pdf/1303.5778.pdf.
// For example, you could implement a bidirectional
// LSTM by using two RNNSeqFunc (forward and backward)
// with two LSTM blocks, and a NetworkSeqFunc for the
// output block.
//
// If the input sequence is of length N, Output will
// be given an input of length N which contains packed
// time steps.
// Each time step fed into Output is packed with the
// forward outputs followed by the backward outputs.
type Bidirectional struct {
	Forward  SeqFunc
	Backward SeqFunc
	Output   SeqFunc
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
	s1, ok1 := slice[0].(SeqFunc)
	s2, ok2 := slice[1].(SeqFunc)
	s3, ok3 := slice[2].(SeqFunc)
	if !ok1 || !ok2 || !ok3 {
		return nil, errors.New("invalid Bidirectional slice types")
	}
	return &Bidirectional{s1, s2, s3}, nil
}

func (b *Bidirectional) BatchSeqs(seqs [][]autofunc.Result) ResultSeqs {
	forwardOut := b.Forward.BatchSeqs(seqs)
	backwardOut := b.Backward.BatchSeqs(reverseInputSeqs(seqs))

	joinedVars := make([][]*autofunc.Variable, len(seqs))
	joinedResults := make([][]autofunc.Result, len(seqs))
	for lane, forwSeq := range forwardOut.OutputSeqs() {
		backSeq := backwardOut.OutputSeqs()[lane]
		joinedSeq := make([]*autofunc.Variable, len(forwSeq))
		joinedRes := make([]autofunc.Result, len(forwSeq))
		for time, forwEntry := range forwSeq {
			backEntry := backSeq[len(forwSeq)-(time+1)]
			fullVec := make(linalg.Vector, len(forwEntry)+len(backEntry))
			copy(fullVec, forwEntry)
			copy(fullVec[len(forwEntry):], backEntry)
			joinedSeq[time] = &autofunc.Variable{Vector: fullVec}
			joinedRes[time] = joinedSeq[time]
		}
		joinedVars[lane] = joinedSeq
		joinedResults[lane] = joinedRes
	}

	return &bidirectionalResult{
		ForwardOut:  forwardOut,
		BackwardOut: backwardOut,
		Joined:      joinedVars,
		Out:         b.Output.BatchSeqs(joinedResults),
	}
}

func (b *Bidirectional) BatchSeqsR(rv autofunc.RVector, seqs [][]autofunc.RResult) RResultSeqs {
	// TODO: this.
	panic("not yet implemented")
}

// Parameters combines the parameters of all three
// internal SeqFuncs, assuming that SeqFuncs have
// no parameters if they are not sgd.Learners.
func (b *Bidirectional) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	for _, x := range []SeqFunc{b.Forward, b.Backward, b.Output} {
		if l, ok := x.(sgd.Learner); ok {
			res = append(res, l.Parameters()...)
		}
	}
	return res
}

func (b *Bidirectional) SerializerType() string {
	return serializerTypeBidirectional
}

// Serialize attempts to serialize b.
// This fails if any of b's internal SeqFuncs are
// not serializer.Serializers.
func (b *Bidirectional) Serialize() ([]byte, error) {
	var slice []serializer.Serializer
	for _, x := range []SeqFunc{b.Forward, b.Backward, b.Output} {
		s, ok := x.(serializer.Serializer)
		if !ok {
			return nil, fmt.Errorf("type cannot be serialized: %T", x)
		}
		slice = append(slice, s)
	}
	return serializer.SerializeSlice(slice)
}

type bidirectionalResult struct {
	ForwardOut  ResultSeqs
	BackwardOut ResultSeqs
	Joined      [][]*autofunc.Variable
	Out         ResultSeqs
}

func (b *bidirectionalResult) OutputSeqs() [][]linalg.Vector {
	return b.Out.OutputSeqs()
}

func (b *bidirectionalResult) Gradient(upstream [][]linalg.Vector, g autofunc.Gradient) {
	for _, joinedSeq := range b.Joined {
		for _, joinedVar := range joinedSeq {
			g[joinedVar] = make(linalg.Vector, len(joinedVar.Vector))
		}
	}

	b.Out.Gradient(upstream, g)

	forwLen := b.forwardOutputSize()
	forwUpstream := make([][]linalg.Vector, len(upstream))
	backUpstream := make([][]linalg.Vector, len(upstream))
	for lane, joinedSeq := range b.Joined {
		subForw := make([]linalg.Vector, len(joinedSeq))
		subBack := make([]linalg.Vector, len(joinedSeq))
		for time, joinedVar := range joinedSeq {
			joinedUpstream := g[joinedVar]
			delete(g, joinedVar)
			subForw[time] = joinedUpstream[:forwLen]
			subBack[len(joinedSeq)-(time+1)] = joinedUpstream[forwLen:]
		}
		forwUpstream[lane] = subForw
		backUpstream[lane] = subBack
	}

	b.ForwardOut.Gradient(forwUpstream, g)
	b.BackwardOut.Gradient(backUpstream, g)
}

func (b *bidirectionalResult) forwardOutputSize() int {
	for _, outSeq := range b.ForwardOut.OutputSeqs() {
		if len(outSeq) > 0 {
			return len(outSeq[0])
		}
	}
	return 0
}

func reverseInputSeqs(seqs [][]autofunc.Result) [][]autofunc.Result {
	res := make([][]autofunc.Result, len(seqs))
	for i, s := range seqs {
		res[i] = make([]autofunc.Result, len(s))
		for j, element := range s {
			res[i][len(s)-(j+1)] = element
		}
	}
	return res
}
