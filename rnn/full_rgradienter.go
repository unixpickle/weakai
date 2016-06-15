package rnn

import (
	"runtime"
	"sort"
	"sync"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/weakai/neuralnet"
)

// FullRGradienter is an RGradienter which computes
// untruncated gradients for batches of sequences.
type FullRGradienter struct {
	Learner       Learner
	CostFunc      neuralnet.CostFunc
	MaxLanes      int
	MaxGoroutines int

	cache       []map[*autofunc.Variable]linalg.Vector
	lastResults []map[*autofunc.Variable]linalg.Vector
}

func (b *FullRGradienter) SeqGradient(seqs []Sequence) autofunc.Gradient {
	res, _ := b.compute(nil, seqs)
	return res
}

func (b *FullRGradienter) SeqRGradient(v autofunc.RVector, seqs []Sequence) (autofunc.Gradient,
	autofunc.RGradient) {
	return b.compute(v, seqs)
}

func (b *FullRGradienter) compute(v autofunc.RVector, seqs []Sequence) (autofunc.Gradient,
	autofunc.RGradient) {
	seqs = sortSeqs(seqs)
	b.freeResults()

	maxGos := b.maxGoroutines()
	if len(seqs) < b.MaxLanes || maxGos == 1 {
		if v == nil {
			return b.syncSeqGradient(seqs), nil
		} else {
			// TODO: this.
			return nil, nil
		}
	}

	batchCount := len(seqs) / b.MaxLanes
	if len(seqs)%b.MaxLanes != 0 {
		batchCount++
	}

	if maxGos > batchCount {
		maxGos = batchCount
	}

	batchChan := make(chan []Sequence, batchCount)
	for i := 0; i < len(seqs); i += b.MaxLanes {
		batchSize := b.MaxLanes
		if batchSize > len(seqs)-i {
			batchSize = len(seqs) - i
		}
		batchChan <- seqs[i : i+batchSize]
	}
	close(batchChan)

	return b.runGoroutines(v, maxGos, batchChan)
}

func (b *FullRGradienter) runGoroutines(v autofunc.RVector, count int,
	batches <-chan []Sequence) (autofunc.Gradient, autofunc.RGradient) {
	var wg sync.WaitGroup
	var gradients []autofunc.Gradient
	var rgradients []autofunc.RGradient
	for i := 0; i < count; i++ {
		grad := b.allocCache()
		gradients = append(gradients, grad)
		var rgrad autofunc.RGradient
		if v != nil {
			rgrad = b.allocCache()
			rgradients = append(rgradients, rgrad)
		}
		go func(g autofunc.Gradient, rg autofunc.RGradient) {
			wg.Done()
			for batch := range batches {
				if rg == nil {
					b.runBatch(g, batch)
				} else {
					// TODO: this.
				}
			}
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
	b.lastResults = append(b.lastResults, gradients[0], rgradients[0])

	return gradients[0], rgradients[0]
}

func (b *FullRGradienter) syncSeqGradient(seqs []Sequence) autofunc.Gradient {
	res := b.allocCache()
	for i := 0; i < len(seqs); i += b.MaxLanes {
		batchSize := b.MaxLanes
		if batchSize > len(seqs)-i {
			batchSize = len(seqs) - i
		}
		b.runBatch(res, seqs[i:i+batchSize])
	}
	b.lastResults = append(b.lastResults, res)
	return res
}

func (b *FullRGradienter) runBatch(g autofunc.Gradient, seqs []Sequence) {
	seqs = removeEmpty(seqs)
	if len(seqs) == 0 {
		return
	}

	emptyState := make(linalg.Vector, b.Learner.StateSize())
	lastStates := make([]linalg.Vector, len(seqs))
	for i := range lastStates {
		lastStates[i] = emptyState
	}

	b.recursiveBatch(g, seqs, lastStates)
}

func (b *FullRGradienter) recursiveBatch(g autofunc.Gradient, seqs []Sequence,
	lastStates []linalg.Vector) []linalg.Vector {
	input := &BlockInput{}
	for lane, seq := range seqs {
		inVar := &autofunc.Variable{Vector: seq.Inputs[0]}
		input.Inputs = append(input.Inputs, inVar)
		inState := &autofunc.Variable{Vector: lastStates[lane]}
		input.States = append(input.States, inState)
	}
	res := b.Learner.Batch(input)

	nextSeqs := removeFirst(seqs)
	var nextStates []linalg.Vector
	for lane, seq := range seqs {
		if len(seq.Inputs) > 1 {
			nextStates = append(nextStates, res.States()[lane])
		}
	}

	upstream := &UpstreamGradient{}
	if len(nextSeqs) != 0 {
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
		outGrad := evalCostFuncDeriv(b.CostFunc, seqs[lane].Outputs[0], output)
		upstream.Outputs = append(upstream.Outputs, outGrad)
	}

	grad := autofunc.Gradient{}
	downstream := make([]linalg.Vector, len(input.States))
	for i, s := range input.States {
		downstream[i] = make(linalg.Vector, b.Learner.StateSize())
		grad[s] = downstream[i]
	}
	res.Gradient(upstream, grad)
	return downstream
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
