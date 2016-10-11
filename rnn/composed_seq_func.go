package rnn

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
)

// ComposedSeqFunc is a SeqFunc that applies multiple
// SeqFuncs in succession.
type ComposedSeqFunc []SeqFunc

// BatchSeqs applies the composed function to a batch of
// input sequences.
func (c ComposedSeqFunc) BatchSeqs(seqs [][]autofunc.Result) ResultSeqs {
	var res composedSeqFuncRes
	for i, x := range c {
		seqRes := x.BatchSeqs(seqs)
		res.results = append(res.results, seqRes)
		if i+1 < len(c) {
			poolVars := make([][]*autofunc.Variable, len(seqRes.OutputSeqs()))
			seqs = make([][]autofunc.Result, len(seqs))
			for j, outSeq := range seqRes.OutputSeqs() {
				poolVars[j] = make([]*autofunc.Variable, len(outSeq))
				seqs[j] = make([]autofunc.Result, len(outSeq))
				for k, outVec := range outSeq {
					poolVars[j][k] = &autofunc.Variable{Vector: outVec}
					seqs[j][k] = poolVars[j][k]
				}
			}
			res.poolVars = append(res.poolVars, poolVars)
		}
	}
	return &res
}

// BatchSeqsR is like BatchSeqs but with support for the
// R-operator.
func (c ComposedSeqFunc) BatchSeqsR(rv autofunc.RVector, seqs [][]autofunc.RResult) RResultSeqs {
	var res composedSeqFuncRRes
	for i, x := range c {
		seqRes := x.BatchSeqsR(rv, seqs)
		res.results = append(res.results, seqRes)
		if i+1 < len(c) {
			poolVars := make([][]*autofunc.Variable, len(seqRes.OutputSeqs()))
			seqs = make([][]autofunc.RResult, len(seqs))
			rOutSeqs := seqRes.ROutputSeqs()
			for j, outSeq := range seqRes.OutputSeqs() {
				poolVars[j] = make([]*autofunc.Variable, len(outSeq))
				seqs[j] = make([]autofunc.RResult, len(outSeq))
				for k, outVec := range outSeq {
					poolVars[j][k] = &autofunc.Variable{Vector: outVec}
					seqs[j][k] = &autofunc.RVariable{
						Variable:   poolVars[j][k],
						ROutputVec: rOutSeqs[j][k],
					}
				}
			}
			res.poolVars = append(res.poolVars, poolVars)
		}
	}
	return &res
}

// Parameters collects all of the parameters from every
// composed SeqFunc that implements sgd.Learner.
func (c ComposedSeqFunc) Parameters() []*autofunc.Variable {
	var res []*autofunc.Variable
	for _, x := range c {
		if learner, ok := x.(sgd.Learner); ok {
			res = append(res, learner.Parameters()...)
		}
	}
	return res
}

type composedSeqFuncRes struct {
	results  []ResultSeqs
	poolVars [][][]*autofunc.Variable
}

func (c *composedSeqFuncRes) OutputSeqs() [][]linalg.Vector {
	lastRes := c.results[len(c.results)-1]
	return lastRes.OutputSeqs()
}

func (c *composedSeqFuncRes) Gradient(upstream [][]linalg.Vector, g autofunc.Gradient) {
	for i := len(c.results) - 1; i >= 0; i-- {
		if i > 0 {
			pool := c.poolVars[i-1]
			for _, poolSeq := range pool {
				for _, poolVar := range poolSeq {
					g[poolVar] = make(linalg.Vector, len(poolVar.Output()))
				}
			}
		}
		c.results[i].Gradient(upstream, g)
		if i > 0 {
			upstream = make([][]linalg.Vector, len(upstream))
			pool := c.poolVars[i-1]
			for j, poolSeq := range pool {
				upstream[j] = make([]linalg.Vector, len(poolSeq))
				for k, poolVar := range poolSeq {
					upstream[j][k] = g[poolVar]
					delete(g, poolVar)
				}
			}
		}
	}
}

type composedSeqFuncRRes struct {
	results  []RResultSeqs
	poolVars [][][]*autofunc.Variable
}

func (c *composedSeqFuncRRes) OutputSeqs() [][]linalg.Vector {
	lastRes := c.results[len(c.results)-1]
	return lastRes.OutputSeqs()
}

func (c *composedSeqFuncRRes) ROutputSeqs() [][]linalg.Vector {
	lastRes := c.results[len(c.results)-1]
	return lastRes.ROutputSeqs()
}

func (c *composedSeqFuncRRes) RGradient(upstream, upstreamR [][]linalg.Vector,
	rg autofunc.RGradient, g autofunc.Gradient) {
	if g == nil {
		// g is used for pooling gradients.
		g = autofunc.Gradient{}
	}
	for i := len(c.results) - 1; i >= 0; i-- {
		if i > 0 {
			pool := c.poolVars[i-1]
			for _, poolSeq := range pool {
				for _, poolVar := range poolSeq {
					g[poolVar] = make(linalg.Vector, len(poolVar.Output()))
					rg[poolVar] = make(linalg.Vector, len(poolVar.Output()))
				}
			}
		}
		c.results[i].RGradient(upstream, upstreamR, rg, g)
		if i > 0 {
			upstream = make([][]linalg.Vector, len(upstream))
			upstreamR = make([][]linalg.Vector, len(upstream))
			pool := c.poolVars[i-1]
			for j, poolSeq := range pool {
				upstream[j] = make([]linalg.Vector, len(poolSeq))
				upstreamR[j] = make([]linalg.Vector, len(poolSeq))
				for k, poolVar := range poolSeq {
					upstream[j][k] = g[poolVar]
					upstreamR[j][k] = rg[poolVar]
					delete(g, poolVar)
					delete(rg, poolVar)
				}
			}
		}
	}
}
