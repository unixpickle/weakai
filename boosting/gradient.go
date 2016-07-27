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

	// Sum is the current ensemble classifier, which
	// is added to during each step of boosting.
	Sum SumClassifier

	// OutCache is the current output of the ensemble
	// classifier.
	// This is used to avoid recomputing the outputs
	// of all the classifiers at each iteration.
	//
	// If you modify Pool, List, Desired, Loss, or Sum
	// during training, you should nil out or recompute
	// CurrentOutput to reflect the new situation.
	OutCache linalg.Vector
}

// Step performs a step of gradient boosting and
// returns the loss before the step was performed.
func (g *Gradient) Step() float64 {
	if g.OutCache == nil {
		g.OutCache = g.Sum.Classify(g.List)
	}
	curOutput := &autofunc.Variable{
		Vector: g.OutCache,
	}
	curLoss := g.Loss.Loss(curOutput, g.Desired)

	grad := autofunc.NewGradient([]*autofunc.Variable{curOutput})
	curLoss.PropagateGradient([]float64{1}, grad)

	classifier := g.Pool.BestClassifier(g.List, grad[curOutput])
	classOutput := classifier.Classify(g.List)
	stepAmount := g.Loss.OptimalStep(curOutput.Vector, classOutput, g.Desired)

	g.Sum.Weights = append(g.Sum.Weights, stepAmount)
	g.Sum.Classifiers = append(g.Sum.Classifiers, classifier)

	g.OutCache.Add(classOutput.Scale(stepAmount))

	return curLoss.Output()[0]
}
