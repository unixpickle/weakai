package seqtoseq

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/seqfunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// A Gradienter is an sgd.RGradienter which works on a
// seqfunc.RFunc for sample sets of Sequence objectg.
//
// After an instance is used once, it should never be
// reused with different parameterg.
type Gradienter struct {
	SeqFunc  seqfunc.RFunc
	Learner  sgd.Learner
	CostFunc neuralnet.CostFunc

	// MaxLanes specifies the maximum number of lanes
	// any BlockInput or BlockRInput may have at once
	// while computing gradientg.
	// If this is 0, a reasonable default is used.
	MaxLanes int

	// MaxGoroutines specifies the maximum number of
	// Goroutines on which to invoke Batch or BatchR
	// on Learner at once.
	MaxGoroutines int

	helper *neuralnet.GradHelper
}

func (g *Gradienter) Gradient(set sgd.SampleSet) autofunc.Gradient {
	return g.makeHelper().Gradient(sortSeqs(set))
}

func (g *Gradienter) RGradient(v autofunc.RVector,
	set sgd.SampleSet) (autofunc.Gradient, autofunc.RGradient) {
	return g.makeHelper().RGradient(v, sortSeqs(set))
}

func (g *Gradienter) makeHelper() *neuralnet.GradHelper {
	if g.helper != nil {
		g.helper.MaxConcurrency = g.MaxGoroutines
		g.helper.MaxSubBatch = g.MaxLanes
		return g.helper
	}
	g.helper = &neuralnet.GradHelper{
		MaxConcurrency: g.MaxGoroutines,
		MaxSubBatch:    g.MaxLanes,
		Learner:        g.Learner,
		CompGrad:       g.runBatch,
		CompRGrad:      g.runBatchR,
	}
	return g.helper
}

func (g *Gradienter) runBatch(grad autofunc.Gradient, set sgd.SampleSet) {
	seqs := sampleSetSlice(set)
	var seqIns [][]linalg.Vector
	for _, s := range seqs {
		seqIns = append(seqIns, s.Inputs)
	}
	output := g.SeqFunc.ApplySeqs(seqfunc.ConstResult(seqIns))

	upstream := make([][]linalg.Vector, len(seqIns))
	for i, outSeq := range output.OutputSeqs() {
		us := make([]linalg.Vector, len(outSeq))
		expectedSeq := seqs[i].Outputs
		for j, actual := range outSeq {
			expected := expectedSeq[j]
			us[j] = costFuncDeriv(g.CostFunc, expected, actual)
		}
		upstream[i] = us
	}

	output.PropagateGradient(upstream, grad)
}

func (g *Gradienter) runBatchR(rv autofunc.RVector, rg autofunc.RGradient,
	grad autofunc.Gradient, set sgd.SampleSet) {
	seqs := sampleSetSlice(set)
	seqIns := make([][]linalg.Vector, len(seqs))
	for _, s := range seqs {
		seqIns = append(seqIns, s.Inputs)
	}
	output := g.SeqFunc.ApplySeqsR(rv, seqfunc.ConstRResult(seqIns))

	upstream := make([][]linalg.Vector, len(seqIns))
	upstreamR := make([][]linalg.Vector, len(seqIns))
	for i, outSeq := range output.OutputSeqs() {
		rOutSeq := output.ROutputSeqs()[i]
		us := make([]linalg.Vector, len(outSeq))
		usR := make([]linalg.Vector, len(outSeq))
		expectedSeq := seqs[i].Outputs
		for j, actual := range outSeq {
			expected := expectedSeq[j]
			us[j], usR[j] = costFuncRDeriv(g.CostFunc, expected, actual, rOutSeq[j])
		}
		upstream[i] = us
		upstreamR[i] = usR
	}

	output.PropagateRGradient(upstream, upstreamR, rg, grad)
}
