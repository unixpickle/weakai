package neuralnet

import (
	"runtime"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// Batcher computes the gradients of neural nets
// for batches of training samples.
//
// It is not safe to call a Batcher's methods
// from multiple Goroutines concurrently.
type Batcher struct {
	costFunc  CostFunc
	batchSize int
	learner   Learner

	reqChan  chan batcherRequest
	respChan chan batcherResponse
}

type batcherRequest struct {
	RV       autofunc.RVector
	Input    linalg.Vector
	Expected linalg.Vector
}

type batcherResponse struct {
	RGrad autofunc.RGradient
	Grad  autofunc.Gradient
}

// NewBatcher creates a Batcher that runs the given
// Learner with the given cost function.
// The Batcher will be optimized for the given
// batchSize, although it will work for other batch
// sizes as well.
// The Batcher will not be running after it is made,
// so you will need to call Start() on ib.
func NewBatcher(l Learner, costFunc CostFunc, batchSize int) *Batcher {
	return &Batcher{
		costFunc:  costFunc,
		batchSize: batchSize,
		learner:   l,
	}
}

// BatchSize returns the optimal batch size for this
// Batcher.
// This is the batch size that was passed to NewBatcher().
func (b *Batcher) BatchSize() int {
	return b.batchSize
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

// Start gets the Batcher ready to compute gradients.
// This is necessary because the Batcher may need to
// launch Goroutines, etc.
func (b *Batcher) Start() {
	routineCount := b.batchSize
	if routineCount > runtime.GOMAXPROCS(0) {
		routineCount = runtime.GOMAXPROCS(0)
	}
	if routineCount < 2 {
		return
	}

	reqChan := make(chan batcherRequest)
	respChan := make(chan batcherResponse, routineCount)
	for i := 0; i < b.batchSize; i++ {
		go func() {
			for req := range reqChan {
				respChan <- b.fulfillRequest(req)
			}
		}()
	}
	b.reqChan = reqChan
	b.respChan = respChan
}

// Stop shuts down any Goroutines that Start() may
// have started.
// You should always call Stop() on a Batcher once
// you are done using ib.
func (b *Batcher) Stop() {
	if b.reqChan != nil {
		close(b.reqChan)
		b.reqChan = nil
		b.respChan = nil
	}
}

// BatchGradient computes the error gradient for a
// batch of samples.
// If the batch is larger than the batchSize passed
// to NewBatcher(), then the gradient will still be
// computed, but not necessarily in an efficient way.
func (b *Batcher) BatchGradient(s *SampleSet) autofunc.Gradient {
	grad, _ := b.batch(nil, s)
	return grad
}

// BatchRGradient computes the Gradient and RGradient
// for a batch of samples.
func (b *Batcher) BatchRGradient(v autofunc.RVector, s *SampleSet) (autofunc.Gradient,
	autofunc.RGradient) {
	return b.batch(v, s)
}

func (b *Batcher) batch(rv autofunc.RVector, s *SampleSet) (autofunc.Gradient,
	autofunc.RGradient) {
	if b.reqChan == nil {
		var gradOut autofunc.Gradient
		var rgradOut autofunc.RGradient
		for i, input := range s.Inputs {
			req := batcherRequest{Input: input, Expected: s.Outputs[i], RV: rv}
			resp := b.fulfillRequest(req)
			if i == 0 {
				gradOut = resp.Grad
				rgradOut = resp.RGrad
			} else {
				gradOut.Add(resp.Grad)
				if rgradOut != nil {
					rgradOut.Add(resp.RGrad)
				}
			}
		}
		return gradOut, rgradOut
	}

	var gradOut autofunc.Gradient
	var rgradOut autofunc.RGradient

	go func() {
		for i, input := range s.Inputs {
			b.reqChan <- batcherRequest{Input: input, Expected: s.Outputs[i], RV: rv}
		}
	}()

	for i := range s.Inputs {
		resp := <-b.respChan
		if i == 0 {
			gradOut = resp.Grad
			rgradOut = resp.RGrad
		} else {
			gradOut.Add(resp.Grad)
			if rgradOut != nil {
				rgradOut.Add(resp.RGrad)
			}
		}
	}

	return gradOut, rgradOut
}

func (b *Batcher) fulfillRequest(req batcherRequest) batcherResponse {
	inVar := &autofunc.Variable{req.Input}
	resp := batcherResponse{Grad: autofunc.NewGradient(b.learner.Parameters())}
	if req.RV != nil {
		resp.RGrad = autofunc.NewRGradient(b.learner.Parameters())
		rVar := autofunc.NewRVariable(inVar, req.RV)
		result := b.learner.ApplyR(req.RV, rVar)
		cost := b.costFunc.CostR(req.RV, req.Expected, result)
		cost.PropagateRGradient(linalg.Vector{1}, linalg.Vector{0},
			resp.RGrad, resp.Grad)
	} else {
		result := b.learner.Apply(inVar)
		cost := b.costFunc.Cost(req.Expected, result)
		cost.PropagateGradient(linalg.Vector{1}, resp.Grad)
	}
	return resp
}
