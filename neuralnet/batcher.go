package neuralnet

import (
	"runtime"
	"sync"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// Batcher computes the gradients of neural nets
// for batches of training samples.
//
// It is not safe to call a Batcher's methods
// from multiple Goroutines concurrently.
type Batcher struct {
	costFunc   CostFunc
	maxThreads int
	learner    Learner

	cache           gradientCache
	lastGradResult  autofunc.Gradient
	lastGradRResult autofunc.RGradient
}

// NewBatcher creates a Batcher that runs the given
// Learner with the given cost function.
// If maxThreads is 0, GOMAXPROCS threads may be used.
func NewBatcher(l Learner, costFunc CostFunc, maxThreads int) *Batcher {
	return &Batcher{
		costFunc:   costFunc,
		maxThreads: maxThreads,
		learner:    l,
		cache:      gradientCache{variables: l.Parameters()},
	}
}

// MaxThreads returns the maximum number of threads this
// Batcher may use.
// This is GOMAXPROCS if 0 was passed to NewBatcher().
func (b *Batcher) MaxThreads() int {
	if b.maxThreads == 0 {
		return runtime.GOMAXPROCS(0)
	}
	return b.maxThreads
}

// Learner returns the learner for which this Batcher
// was created.
func (b *Batcher) Learner() Learner {
	return b.learner
}

// CostFunc returns the CostFunc that this Batcher
// uses to compute gradients.
func (b *Batcher) CostFunc() CostFunc {
	return b.costFunc
}

// BatchGradient computes the error gradient for a
// batch of samples.
// If the batch is larger than the batchSize passed
// to NewBatcher(), then the gradient will still be
// computed, but not necessarily in an efficient way.
//
// The resulting values are  only valid until the next
// call to BatchGradient or BatchRGradient, when the
// vectors may be re-used.
func (b *Batcher) BatchGradient(s *SampleSet) autofunc.Gradient {
	grad, _ := b.batch(nil, s)
	return grad
}

// BatchRGradient computes the Gradient and RGradient
// for a batch of samples.
//
// The resulting values are  only valid until the next
// call to BatchGradient or BatchRGradient, when the
// vectors may be re-used.
func (b *Batcher) BatchRGradient(v autofunc.RVector, s *SampleSet) (autofunc.Gradient,
	autofunc.RGradient) {
	return b.batch(v, s)
}

func (b *Batcher) batch(rv autofunc.RVector, s *SampleSet) (autofunc.Gradient,
	autofunc.RGradient) {
	if b.lastGradResult != nil {
		b.cache.Free(b.lastGradResult)
	}
	if b.lastGradRResult != nil {
		b.cache.FreeR(b.lastGradRResult)
	}

	sampleIdxChan := make(chan int, len(s.Inputs))
	for i := range s.Inputs {
		sampleIdxChan <- i
	}
	close(sampleIdxChan)

	var wg sync.WaitGroup
	var grads []autofunc.Gradient
	var rgrads []autofunc.RGradient

	routineCount := b.MaxThreads()
	if len(s.Inputs) < routineCount {
		routineCount = len(s.Inputs)
	}

	for i := 0; i < routineCount; i++ {
		grad := b.cache.Alloc()
		var rgrad autofunc.RGradient
		if rv != nil {
			rgrad = b.cache.AllocR()
		}
		if routineCount > 1 {
			wg.Add(1)
			go func(grad autofunc.Gradient, rgrad autofunc.RGradient) {
				defer wg.Done()
				for sampleIdx := range sampleIdxChan {
					input, output := s.Inputs[sampleIdx], s.Outputs[sampleIdx]
					b.addGrads(grad, rgrad, rv, input, output)
				}
			}(grad, rgrad)
		} else {
			for sampleIdx := range sampleIdxChan {
				input, output := s.Inputs[sampleIdx], s.Outputs[sampleIdx]
				b.addGrads(grad, rgrad, rv, input, output)
			}
		}
		grads = append(grads, grad)
		if rv != nil {
			rgrads = append(rgrads, rgrad)
		}
	}

	if routineCount > 1 {
		wg.Wait()
	}

	for i := 1; i < len(grads); i++ {
		grads[0].Add(grads[i])
		b.cache.Free(grads[i])
	}
	for i := 1; i < len(rgrads); i++ {
		rgrads[0].Add(rgrads[i])
		b.cache.FreeR(rgrads[i])
	}

	b.lastGradResult = grads[0]
	if rgrads != nil {
		b.lastGradRResult = rgrads[0]
	}

	return b.lastGradResult, b.lastGradRResult
}

func (b *Batcher) addGrads(grad autofunc.Gradient, rgrad autofunc.RGradient,
	rv autofunc.RVector, input, expected linalg.Vector) {
	inVar := &autofunc.Variable{input}
	if rgrad != nil {
		rVar := autofunc.NewRVariable(inVar, rv)
		result := b.learner.ApplyR(rv, rVar)
		cost := b.costFunc.CostR(rv, expected, result)
		cost.PropagateRGradient(linalg.Vector{1}, linalg.Vector{0},
			rgrad, grad)
		cost.Release()
	} else {
		result := b.learner.Apply(inVar)
		cost := b.costFunc.Cost(expected, result)
		cost.PropagateGradient(linalg.Vector{1}, grad)
		cost.Release()
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
	res := g.rGradients[len(g.gradients)-1]
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
