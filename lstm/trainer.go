package lstm

import (
	"math/rand"

	"github.com/unixpickle/num-analysis/linalg"
)

type Trainer struct {
	InSeqs  [][]linalg.Vector
	OutSeqs [][]linalg.Vector

	CostFunc CostFunc

	StepSize float64
	Epochs   int
}

func (t *Trainer) Train(r *RNN) {
	for i := 0; i < t.Epochs; i++ {
		perm := rand.Perm(len(t.InSeqs))
		for _, j := range perm {
			seq := t.InSeqs[j]
			outSeq := t.OutSeqs[j]
			var costPartials []linalg.Vector
			for k, input := range seq {
				actualOut := r.StepTime(input)
				costGrad := make(linalg.Vector, len(actualOut))
				t.CostFunc.Gradient(actualOut, outSeq[k], costGrad)
				costPartials = append(costPartials, costGrad)
			}
			grad := r.CostGradient(costPartials)
			grad.Scale(-t.StepSize)
			r.StepGradient(grad)
		}
	}
}
