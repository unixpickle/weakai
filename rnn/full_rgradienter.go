package rnn

import (
	"runtime"
	"sort"
	"sync"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

const defaultMaxLanes = 10

// FullRGradienter is an RGradienter which computes
// untruncated gradients for sets of Sequences.
//
// After a FullRGradienter is used on a BatchLearner,
// it should never be reused on any BatchLearner with
// a different set of parameters.
type FullRGradienter struct {
	Learner  BlockLearner
	CostFunc neuralnet.CostFunc

	// MaxLanes specifies the maximum number of lanes
	// any BlockInput or BlockRInput may have at once
	// while computing gradients.
	// If this is 0, a reasonable default is used.
	MaxLanes int

	// MaxGoroutines specifies the maximum number of
	// Goroutines on which to invoke Batch or BatchR
	// on Learner at once.
	MaxGoroutines int

	cache       []map[*autofunc.Variable]linalg.Vector
	lastResults []map[*autofunc.Variable]linalg.Vector
}

func (b *FullRGradienter) Gradient(s neuralnet.SampleSet) autofunc.Gradient {
	res, _ := b.compute(nil, sampleSetSequences(s))
	return res
}

func (b *FullRGradienter) RGradient(v autofunc.RVector,
	s neuralnet.SampleSet) (autofunc.Gradient, autofunc.RGradient) {
	return b.compute(v, sampleSetSequences(s))
}

func (b *FullRGradienter) compute(v autofunc.RVector, seqs []Sequence) (autofunc.Gradient,
	autofunc.RGradient) {
	seqs = sortSeqs(seqs)
	b.freeResults()

	maxGos := b.maxGoroutines()
	if len(seqs) < b.maxLanes() || maxGos == 1 {
		return b.syncGradient(v, seqs)
	}

	batchCount := len(seqs) / b.maxLanes()
	if len(seqs)%b.maxLanes() != 0 {
		batchCount++
	}

	if maxGos > batchCount {
		maxGos = batchCount
	}

	batchChan := make(chan []Sequence, batchCount)
	for i := 0; i < len(seqs); i += b.maxLanes() {
		batchSize := b.maxLanes()
		if batchSize > len(seqs)-i {
			batchSize = len(seqs) - i
		}
		batchChan <- seqs[i : i+batchSize]
	}
	close(batchChan)

	return b.asyncGradient(v, maxGos, batchChan)
}

func (b *FullRGradienter) asyncGradient(v autofunc.RVector, count int,
	batches <-chan []Sequence) (autofunc.Gradient, autofunc.RGradient) {
	var wg sync.WaitGroup
	var gradients []autofunc.Gradient
	var rgradients []autofunc.RGradient
	for i := 0; i < count; i++ {
		wg.Add(1)
		grad := b.allocCache()
		gradients = append(gradients, grad)
		var rgrad autofunc.RGradient
		if v != nil {
			rgrad = b.allocCache()
			rgradients = append(rgradients, rgrad)
		}
		go func(g autofunc.Gradient, rg autofunc.RGradient) {
			for batch := range batches {
				b.runBatch(v, g, rg, batch)
			}
			wg.Done()
		}(grad, rgrad)
	}
	wg.Wait()

	for i := 1; i < len(gradients); i++ {
		gradients[0].Add(gradients[i])
		b.cache = append(b.cache, gradients[i])
		if rgradients != nil {
			rgradients[0].Add(rgradients[i])
			b.cache = append(b.cache, rgradients[i])
		}
	}

	b.lastResults = append(b.lastResults, gradients[0])
	if rgradients != nil {
		b.lastResults = append(b.lastResults, rgradients[0])
		return gradients[0], rgradients[0]
	} else {
		return gradients[0], nil
	}
}

func (b *FullRGradienter) syncGradient(v autofunc.RVector,
	seqs []Sequence) (grad autofunc.Gradient, rgrad autofunc.RGradient) {
	grad = b.allocCache()
	if v != nil {
		rgrad = b.allocCache()
	}
	for i := 0; i < len(seqs); i += b.maxLanes() {
		batchSize := b.maxLanes()
		if batchSize > len(seqs)-i {
			batchSize = len(seqs) - i
		}
		b.runBatch(v, grad, rgrad, seqs[i:i+batchSize])
	}
	b.lastResults = append(b.lastResults, grad)
	if rgrad != nil {
		b.lastResults = append(b.lastResults, rgrad)
	}
	return
}

func (b *FullRGradienter) runBatch(v autofunc.RVector, g autofunc.Gradient,
	rg autofunc.RGradient, seqs []Sequence) {
	seqs = removeEmpty(seqs)
	if len(seqs) == 0 {
		return
	}

	emptyState := make(linalg.Vector, b.Learner.StateSize())
	zeroStates := make([]linalg.Vector, len(seqs))
	for i := range zeroStates {
		zeroStates[i] = emptyState
	}

	if v != nil {
		b.recursiveRBatch(v, g, rg, seqs, zeroStates, zeroStates)
	} else {
		b.recursiveBatch(g, seqs, zeroStates)
	}
}

func (b *FullRGradienter) recursiveBatch(g autofunc.Gradient, seqs []Sequence,
	lastStates []linalg.Vector) []linalg.Vector {
	input := seqHeadInput(seqs, lastStates)
	res := b.Learner.Batch(input)

	// Compute upstream state derivatives recursively.
	upstream := &UpstreamGradient{}
	nextSeqs := removeFirst(seqs)
	if len(nextSeqs) != 0 {
		var nextStates []linalg.Vector
		for lane, seq := range seqs {
			if len(seq.Inputs) > 1 {
				nextStates = append(nextStates, res.States()[lane])
			}
		}
		res := b.recursiveBatch(g, nextSeqs, nextStates)
		var resIdx int
		for _, seq := range seqs {
			if len(seq.Inputs) > 1 {
				upstream.States = append(upstream.States, res[resIdx])
				resIdx++
			} else {
				emptyState := make(linalg.Vector, b.Learner.StateSize())
				upstream.States = append(upstream.States, emptyState)
			}
		}
	}

	for lane, output := range res.Outputs() {
		outGrad := costFuncDeriv(b.CostFunc, seqs[lane].Outputs[0], output)
		upstream.Outputs = append(upstream.Outputs, outGrad)
	}

	// Compute downstream state derivatives & back propagate.
	downstream := make([]linalg.Vector, len(input.States))
	for i, s := range input.States {
		downstream[i] = make(linalg.Vector, b.Learner.StateSize())
		g[s] = downstream[i]
	}
	res.Gradient(upstream, g)
	for _, s := range input.States {
		delete(g, s)
	}
	return downstream
}

func (b *FullRGradienter) recursiveRBatch(v autofunc.RVector, g autofunc.Gradient,
	rg autofunc.RGradient, seqs []Sequence, states, rStates []linalg.Vector) (stateGrad,
	stateRGrad []linalg.Vector) {
	input := seqHeadRInput(seqs, states, rStates)
	res := b.Learner.BatchR(v, input)

	// Compute upstream state derivatives recursively.
	upstream := &UpstreamRGradient{}
	nextSeqs := removeFirst(seqs)
	if len(nextSeqs) != 0 {
		var nextStates []linalg.Vector
		var nextRStates []linalg.Vector
		for lane, seq := range seqs {
			if len(seq.Inputs) > 1 {
				nextStates = append(nextStates, res.States()[lane])
				nextRStates = append(nextRStates, res.RStates()[lane])
			}
		}
		states, statesR := b.recursiveRBatch(v, g, rg, nextSeqs, nextStates, nextRStates)
		var stateIdx int
		for _, seq := range seqs {
			if len(seq.Inputs) > 1 {
				upstream.States = append(upstream.States, states[stateIdx])
				upstream.RStates = append(upstream.RStates, statesR[stateIdx])
				stateIdx++
			} else {
				emptyState := make(linalg.Vector, b.Learner.StateSize())
				upstream.States = append(upstream.States, emptyState)
				upstream.RStates = append(upstream.RStates, emptyState)
			}
		}
	}

	for lane, output := range res.Outputs() {
		rOutput := res.ROutputs()[lane]
		outGrad, outRGrad := costFuncRDeriv(v, b.CostFunc, seqs[lane].Outputs[0],
			output, rOutput)
		upstream.Outputs = append(upstream.Outputs, outGrad)
		upstream.ROutputs = append(upstream.ROutputs, outRGrad)
	}

	// Compute downstream state derivatives & back propagate.
	stateGrad = make([]linalg.Vector, len(input.States))
	stateRGrad = make([]linalg.Vector, len(input.States))
	for i, s := range input.States {
		stateGrad[i] = make(linalg.Vector, b.Learner.StateSize())
		stateRGrad[i] = make(linalg.Vector, b.Learner.StateSize())
		g[s.Variable] = stateGrad[i]
		rg[s.Variable] = stateRGrad[i]
	}
	res.RGradient(upstream, rg, g)
	for _, s := range input.States {
		delete(g, s.Variable)
		delete(rg, s.Variable)
	}
	return
}

func (b *FullRGradienter) allocCache() map[*autofunc.Variable]linalg.Vector {
	if len(b.cache) == 0 {
		return autofunc.NewGradient(b.Learner.Parameters())
	} else {
		res := b.cache[len(b.cache)-1]
		autofunc.Gradient(res).Zero()
		b.cache = b.cache[:len(b.cache)-1]
		return res
	}
}

func (b *FullRGradienter) freeResults() {
	b.cache = append(b.cache, b.lastResults...)
	b.lastResults = nil
}

func (b *FullRGradienter) maxGoroutines() int {
	if b.MaxGoroutines == 0 {
		return runtime.GOMAXPROCS(0)
	} else {
		return b.MaxGoroutines
	}
}

func (b *FullRGradienter) maxLanes() int {
	if b.MaxLanes == 0 {
		return defaultMaxLanes
	} else {
		return b.MaxLanes
	}
}

func removeEmpty(seqs []Sequence) []Sequence {
	var res []Sequence
	for _, seq := range seqs {
		if len(seq.Inputs) != 0 {
			res = append(res, seq)
		}
	}
	return res
}

func removeFirst(seqs []Sequence) []Sequence {
	var nextSeqs []Sequence
	for _, seq := range seqs {
		if len(seq.Inputs) == 1 {
			continue
		}
		s := Sequence{Inputs: seq.Inputs[1:], Outputs: seq.Outputs[1:]}
		nextSeqs = append(nextSeqs, s)
	}
	return nextSeqs
}

type seqSorter []Sequence

func sortSeqs(s []Sequence) []Sequence {
	res := make(seqSorter, len(s))
	copy(res, s)
	sort.Sort(res)
	return res
}

func (s seqSorter) Len() int {
	return len(s)
}

func (s seqSorter) Less(i, j int) bool {
	return len(s[i].Inputs) < len(s[j].Inputs)
}

func (s seqSorter) Swap(i, j int) {
	s[i], s[j] = s[j], s[i]
}
