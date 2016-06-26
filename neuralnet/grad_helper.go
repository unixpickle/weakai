package neuralnet

import (
	"runtime"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/sgd"
)

const defaultMaxSubBatch = 15

type gradResult struct {
	Grad  autofunc.Gradient
	RGrad autofunc.RGradient
}

// A GradHelper is an RGradienter which runs gradient
// computations in parallel and divides up SampleSets
// as needed.
type GradHelper struct {
	// MaxConcurrency is the maximum number of goroutines
	// on which the underlying gradient functions can be
	// run at once.
	// If this is 0, GOMAXPROCS is used.
	MaxConcurrency int

	// MaxSubBatch is the maximum number of samples that
	// can be passed to the underlying gradient functions
	// in one call.
	// If this is 0, a reasonable default is used.
	MaxSubBatch int

	// Learner provides the GradHelper with a list of
	// parameters so that it can allocate and cache
	// gradient vectors.
	// The Learner and its parameters should not change
	// once you start using the GradHelper, since it
	// caches gradients and assumes static parameters.
	Learner sgd.Learner

	CompGrad  func(g autofunc.Gradient, s sgd.SampleSet)
	CompRGrad func(rv autofunc.RVector, rg autofunc.RGradient,
		g autofunc.Gradient, s sgd.SampleSet)

	gradCache       gradientCache
	lastGradResult  autofunc.Gradient
	lastRGradResult autofunc.RGradient
}

func (g *GradHelper) Gradient(s sgd.SampleSet) autofunc.Gradient {
	grad, _ := g.batch(nil, s)
	return grad
}

func (g *GradHelper) RGradient(rv autofunc.RVector, s sgd.SampleSet) (autofunc.Gradient,
	autofunc.RGradient) {
	return g.batch(rv, s)
}

func (g *GradHelper) batch(rv autofunc.RVector, s sgd.SampleSet) (grad autofunc.Gradient,
	rgrad autofunc.RGradient) {
	g.gradCache.variables = g.Learner.Parameters()
	if g.lastGradResult != nil {
		g.gradCache.Free(g.lastGradResult)
	}
	if g.lastRGradResult != nil {
		g.gradCache.FreeR(g.lastRGradResult)
	}
	batchSize := g.batchSize()
	maxGos := g.goroutineCount()
	if s.Len() < batchSize || maxGos < 2 {
		grad, rgrad = g.runSync(rv, s)
	} else {
		grad, rgrad = g.runAsync(rv, s)
	}
	g.lastGradResult = grad
	g.lastRGradResult = rgrad
	return
}

func (g *GradHelper) runSync(rv autofunc.RVector, s sgd.SampleSet) (grad autofunc.Gradient,
	rgrad autofunc.RGradient) {
	grad = g.gradCache.Alloc()
	if rv != nil {
		rgrad = g.gradCache.AllocR()
	}
	for subset := range g.subBatches(s) {
		if rv != nil {
			g.CompRGrad(rv, rgrad, grad, subset)
		} else {
			g.CompGrad(grad, subset)
		}
	}
	return
}

func (g *GradHelper) runAsync(rv autofunc.RVector, s sgd.SampleSet) (grad autofunc.Gradient,
	rgrad autofunc.RGradient) {
	inChan := g.subBatches(s)
	resChan := make(chan gradResult)

	goCount := g.goroutineCount()
	for i := 0; i < goCount; i++ {
		subGrad := g.gradCache.Alloc()
		var subRGrad autofunc.RGradient
		if rv != nil {
			subRGrad = g.gradCache.AllocR()
		}
		go func(subGrad autofunc.Gradient, subRGrad autofunc.RGradient) {
			for subset := range inChan {
				if rv != nil {
					g.CompRGrad(rv, subRGrad, subGrad, subset)
				} else {
					g.CompGrad(subGrad, subset)
				}
			}
			resChan <- gradResult{subGrad, subRGrad}
		}(subGrad, subRGrad)
	}

	for i := 0; i < goCount; i++ {
		res := <-resChan
		if grad == nil {
			grad = res.Grad
			rgrad = res.RGrad
		} else {
			grad.Add(res.Grad)
			g.gradCache.Free(res.Grad)
			if res.RGrad != nil {
				rgrad.Add(res.RGrad)
				g.gradCache.FreeR(res.RGrad)
			}
		}
	}

	return
}

func (g *GradHelper) subBatches(s sgd.SampleSet) <-chan sgd.SampleSet {
	batchSize := g.batchSize()
	res := make(chan sgd.SampleSet, s.Len()/batchSize+1)
	for i := 0; i < s.Len(); i += batchSize {
		bs := batchSize
		if bs > s.Len()-i {
			bs = s.Len() - i
		}
		res <- s.Subset(i, i+bs)
	}
	close(res)
	return res
}

func (g *GradHelper) goroutineCount() int {
	max := runtime.GOMAXPROCS(0)
	if g.MaxConcurrency == 0 || g.MaxConcurrency > max {
		return max
	} else {
		return g.MaxConcurrency
	}
}

func (g *GradHelper) batchSize() int {
	if g.MaxSubBatch != 0 {
		return g.MaxSubBatch
	} else {
		return defaultMaxSubBatch
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
