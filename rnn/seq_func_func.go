package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// SeqFuncFunc wraps a SeqFunc and exposes an Apply,
// ApplyR, Batch, and BatchR method, making the SeqFunc
// look like an function on vectors rather than on
// variable-length sequences.
type SeqFuncFunc struct {
	S SeqFunc

	// InSize is the size of the input vectors that S
	// takes at each timestep.
	InSize int
}

// Apply splits the input vector into timesteps of length
// s.InSize and applies the s.S to the result.
// It returns a concatenated form of the output sequence.
//
// The result will always report that it is not constant,
// even if it depends on no gradient variables. This is
// due to the fact that ResultSeqs does not report if it
// is a constant or not.
func (s *SeqFuncFunc) Apply(in autofunc.Result) autofunc.Result {
	return s.Batch(in, 1)
}

// ApplyR is like Apply but for autofunc.RResults.
func (s *SeqFuncFunc) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return s.BatchR(v, in, 1)
}

// Batch is equivalent to splitting the input up into n
// input sequence vectors, running Apply on each of them,
// and joining the results.
func (s *SeqFuncFunc) Batch(in autofunc.Result, n int) autofunc.Result {
	res := s.S.BatchSeqs(splitSeqBatch(in, n, s.InSize))
	return &seqFuncFuncResult{
		res:    res,
		joined: joinSeqBatchVec(res.OutputSeqs()),
	}
}

// BatchR is like Batch but with autofunc.RResults.
func (s *SeqFuncFunc) BatchR(rv autofunc.RVector, in autofunc.RResult, n int) autofunc.RResult {
	res := s.S.BatchSeqsR(rv, splitSeqBatchR(in, n, s.InSize))
	return &seqFuncFuncRResult{
		res:     res,
		joined:  joinSeqBatchVec(res.OutputSeqs()),
		joinedR: joinSeqBatchVec(res.ROutputSeqs()),
	}
}

type seqFuncFuncResult struct {
	res    ResultSeqs
	joined linalg.Vector
}

func (s *seqFuncFuncResult) Output() linalg.Vector {
	return s.joined
}

func (s *seqFuncFuncResult) Constant(g autofunc.Gradient) bool {
	return false
}

func (s *seqFuncFuncResult) PropagateGradient(upstream linalg.Vector, g autofunc.Gradient) {
	n, inSize := seqDims(s.res.OutputSeqs())
	us := splitSeqBatchVec(upstream, n, inSize)
	s.res.Gradient(us, g)
}

type seqFuncFuncRResult struct {
	res     RResultSeqs
	joined  linalg.Vector
	joinedR linalg.Vector
}

func (s *seqFuncFuncRResult) Output() linalg.Vector {
	return s.joined
}

func (s *seqFuncFuncRResult) ROutput() linalg.Vector {
	return s.joinedR
}

func (s *seqFuncFuncRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return false
}

func (s *seqFuncFuncRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rg autofunc.RGradient, g autofunc.Gradient) {
	n, inSize := seqDims(s.res.OutputSeqs())
	us := splitSeqBatchVec(upstream, n, inSize)
	usR := splitSeqBatchVec(upstreamR, n, inSize)
	s.res.RGradient(us, usR, rg, g)
}

func joinSeqBatchVec(seqs [][]linalg.Vector) linalg.Vector {
	var totalSize int
	for _, seq := range seqs {
		for _, ts := range seq {
			totalSize += len(ts)
		}
	}

	res := make(linalg.Vector, totalSize)
	var idx int
	for _, seq := range seqs {
		for _, ts := range seq {
			copy(res[idx:], ts)
			idx += len(ts)
		}
	}
	return res
}

func seqDims(outputs [][]linalg.Vector) (count, inSize int) {
	count = len(outputs)
	if count > 0 {
		if len(outputs[0]) > 0 {
			inSize = len(outputs[0][0])
		}
	}
	return
}

func splitSeqBatchVec(batch linalg.Vector, n, inSize int) [][]linalg.Vector {
	timeSteps := len(batch) / (n * inSize)
	var seqs [][]linalg.Vector
	var idx int
	for i := 0; i < n; i++ {
		var seq []linalg.Vector
		for j := 0; j < timeSteps; j++ {
			ts := batch[idx : idx+inSize]
			idx += inSize
			seq = append(seq, ts)
		}
		seqs = append(seqs, seq)
	}
	return seqs
}

func splitSeqBatch(batch autofunc.Result, n, inSize int) [][]autofunc.Result {
	timeSteps := len(batch.Output()) / (n * inSize)
	var seqs [][]autofunc.Result
	var idx int
	for i := 0; i < n; i++ {
		var seq []autofunc.Result
		for j := 0; j < timeSteps; j++ {
			ts := autofunc.Slice(batch, idx, idx+inSize)
			idx += inSize
			seq = append(seq, ts)
		}
		seqs = append(seqs, seq)
	}
	return seqs
}

func splitSeqBatchR(batch autofunc.RResult, n, inSize int) [][]autofunc.RResult {
	timeSteps := len(batch.Output()) / (n * inSize)
	var seqs [][]autofunc.RResult
	var idx int
	for i := 0; i < n; i++ {
		var seq []autofunc.RResult
		for j := 0; j < timeSteps; j++ {
			ts := autofunc.SliceR(batch, idx, idx+inSize)
			idx += inSize
			seq = append(seq, ts)
		}
		seqs = append(seqs, seq)
	}
	return seqs
}
