package neuralnet

import (
	"runtime"
	"sync"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

const defaultMaxBatchSize = 10

type gradResult struct {
	Grad  autofunc.Gradient
	RGrad autofunc.RGradient
}

// BatchRGradienter is an RGradienter that computes
// the gradients of BatchLearners using SampleSets
// of VectorSamples.
// It is not safe to call its methods concurrently
// from multiple Goroutines at once.
//
// A BatchRGradienter is suitable for training tasks
// with lots of learnable parameters.
// Such tasks can benefit from parallelism, both
// from using the BatchLearner's batching features
// and from Goroutine-level concurrency.
//
// After you use a BatchRGradienter with a given
// BatchLearner, you should never use the same
// BatchRGradienter for any BatchLearner with
// different parameters.
type BatchRGradienter struct {
	Learner  BatchLearner
	CostFunc CostFunc

	// MaxGoroutines is the maximum number of Goroutines
	// the BatchRGradienter will use simultaneously.
	// If this is 0, a reasonable default is used.
	MaxGoroutines int

	// MaxBatchSize is the maximum number of samples the
	// BatchRGradienter will pass to the learner at once.
	// If this is 0, a reasonable default is used.
	MaxBatchSize int

	gradCache       gradientCache
	lastGradResult  autofunc.Gradient
	lastGradRResult autofunc.RGradient
}

func (b *BatchRGradienter) Gradient(s SampleSet) autofunc.Gradient {
	grad, _ := b.batch(nil, s)
	return grad
}

func (b *BatchRGradienter) RGradient(v autofunc.RVector, s SampleSet) (autofunc.Gradient,
	autofunc.RGradient) {
	return b.batch(v, s)
}

func (b *BatchRGradienter) batch(rv autofunc.RVector, s SampleSet) (autofunc.Gradient,
	autofunc.RGradient) {
	if b.gradCache.variables == nil {
		b.gradCache.variables = b.Learner.Parameters()
	} else {
		if b.lastGradResult != nil {
			b.gradCache.Free(b.lastGradResult)
		}
		if b.lastGradRResult != nil {
			b.gradCache.FreeR(b.lastGradRResult)
		}
	}

	maxGos := b.goroutineCount()
	maxBatch := b.batchSize()

	if maxGos < 2 || s.Len() <= maxBatch {
		b.lastGradResult, b.lastGradRResult = b.runBatches(rv, s)
		return b.lastGradResult, b.lastGradRResult
	}

	goCount := s.Len() / maxBatch
	if s.Len()%maxBatch != 0 {
		goCount++
	}
	if goCount > maxGos {
		goCount = maxGos
	}

	batchChan := make(chan SampleSet, s.Len()/maxBatch+1)
	for i := 0; i < s.Len(); i += maxBatch {
		bs := maxBatch
		if bs > s.Len()-i {
			bs = s.Len() - i
		}
		batchChan <- s.Subset(i, i+bs)
	}
	close(batchChan)

	resChan := b.launchGoroutines(rv, batchChan, goCount)

	b.lastGradResult = nil
	b.lastGradRResult = nil
	for res := range resChan {
		if b.lastGradResult == nil {
			b.lastGradResult = res.Grad
		} else {
			b.lastGradResult.Add(res.Grad)
			b.gradCache.Free(res.Grad)
		}
		if b.lastGradRResult == nil {
			b.lastGradRResult = res.RGrad
		} else {
			b.lastGradRResult.Add(res.RGrad)
			b.gradCache.FreeR(res.RGrad)
		}
	}

	return b.lastGradResult, b.lastGradRResult
}

func (b *BatchRGradienter) runBatches(rv autofunc.RVector, s SampleSet) (autofunc.Gradient,
	autofunc.RGradient) {
	grad := b.gradCache.Alloc()
	var rgrad autofunc.RGradient
	if rv != nil {
		rgrad = b.gradCache.AllocR()
	}

	batchSize := b.batchSize()
	for i := 0; i < s.Len(); i += batchSize {
		bs := batchSize
		if bs > s.Len()-i {
			bs = s.Len() - i
		}
		b.runBatch(rv, s.Subset(i, i+bs), grad, rgrad)
	}

	return grad, rgrad
}

func (b *BatchRGradienter) launchGoroutines(rv autofunc.RVector,
	in <-chan SampleSet, goCount int) <-chan gradResult {
	resChan := make(chan gradResult)
	var wg sync.WaitGroup
	for i := 0; i < goCount; i++ {
		wg.Add(1)
		grad := b.gradCache.Alloc()
		var rgrad autofunc.RGradient
		if rv != nil {
			rgrad = b.gradCache.AllocR()
		}
		go func(grad autofunc.Gradient, rgrad autofunc.RGradient) {
			defer wg.Done()
			for batch := range in {
				b.runBatch(rv, batch, grad, rgrad)
			}
			resChan <- gradResult{grad, rgrad}
		}(grad, rgrad)
	}
	go func() {
		wg.Wait()
		close(resChan)
	}()
	return resChan
}

func (b *BatchRGradienter) runBatch(rv autofunc.RVector, s SampleSet, grad autofunc.Gradient,
	rgrad autofunc.RGradient) {
	if s.Len() == 0 {
		return
	}
	sampleCount := s.Len()
	firstSample := s.GetSample(0).(VectorSample)
	inputSize := len(firstSample.Input)
	outputSize := len(firstSample.Output)
	inVec := make(linalg.Vector, sampleCount*inputSize)
	outVec := make(linalg.Vector, sampleCount*outputSize)

	for i := 0; i < s.Len(); i++ {
		sample := s.GetSample(i)
		vs := sample.(VectorSample)
		copy(inVec[i*inputSize:], vs.Input)
		copy(outVec[i*outputSize:], vs.Output)
	}

	inVar := &autofunc.Variable{inVec}
	if rgrad != nil {
		rVar := autofunc.NewRVariable(inVar, rv)
		result := b.Learner.BatchR(rv, rVar, sampleCount)
		cost := b.CostFunc.CostR(rv, outVec, result)
		cost.PropagateRGradient(linalg.Vector{1}, linalg.Vector{0},
			rgrad, grad)
	} else {
		result := b.Learner.Batch(inVar, sampleCount)
		cost := b.CostFunc.Cost(outVec, result)
		cost.PropagateGradient(linalg.Vector{1}, grad)
	}
}

func (b *BatchRGradienter) goroutineCount() int {
	max := runtime.GOMAXPROCS(0)
	if b.MaxGoroutines == 0 || b.MaxGoroutines > max {
		return max
	} else {
		return b.MaxGoroutines
	}
}

func (b *BatchRGradienter) batchSize() int {
	if b.MaxBatchSize != 0 {
		return b.MaxBatchSize
	} else {
		return defaultMaxBatchSize
	}
}

type gradientCache struct {
	variables  []*autofunc.Variable
	gradients  []autofunc.Gradient
	rGradients []autofunc.RGradient
}

func (g *gradientCache) Alloc() autofunc.Gradient {
	if len(g.gradients) == 0 {
		res := autofunc.NewGradient(g.variables)
		return res
	}
	res := g.gradients[len(g.gradients)-1]
	g.gradients = g.gradients[:len(g.gradients)-1]
	res.Zero()
	return res
}

func (g *gradientCache) AllocR() autofunc.RGradient {
	if len(g.rGradients) == 0 {
		res := autofunc.NewRGradient(g.variables)
		return res
	}
	res := g.rGradients[len(g.rGradients)-1]
	g.rGradients = g.rGradients[:len(g.rGradients)-1]
	res.Zero()
	return res
}

func (g *gradientCache) Free(gr autofunc.Gradient) {
	g.gradients = append(g.gradients, gr)
}

func (g *gradientCache) FreeR(gr autofunc.RGradient) {
	g.rGradients = append(g.rGradients, gr)
}
