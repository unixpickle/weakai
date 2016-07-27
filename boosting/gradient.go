package boosting

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// Gradient performs gradient boosting.
type Gradient struct {
	// Loss is the loss function to use for training.
	Loss LossFunc

	// Desired is the vector of desired classifications.
	Desired linalg.Vector

	// List is the list of samples to boost on.
	List SampleList

	// Pool is used to create weak learners for training.
	Pool Pool

	curSum SumClassifier
}

// Solution returns the current ensemble classifier.
func (g *Gradient) Solution() *SumClassifier {
	return &g.curSum
}

// Step performs a step of gradient boosting and
// returns the loss before the step was performed.
func (g *Gradient) Step() float64 {
	curOutput := &autofunc.Variable{
		Vector: g.curSum.Classify(g.List),
	}
	curLoss := g.Loss.Loss(curOutput, g.Desired)

	grad := autofunc.NewGradient([]*autofunc.Variable{curOutput})
	curLoss.PropagateGradient([]float64{1}, grad)

	classifier := g.Pool.BestClassifier(g.List, grad[curOutput])
	classOutput := classifier.Classify(g.List)
	stepAmount := g.Loss.OptimalStep(curOutput.Vector, classOutput, g.Desired)

	g.curSum.Weights = append(g.curSum.Weights, stepAmount)
	g.curSum.Classifiers = append(g.curSum.Classifiers, classifier)

	return curLoss.Output()[0]
}
