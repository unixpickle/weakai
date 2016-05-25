package rbm

import "github.com/unixpickle/num-analysis/linalg"

// RBMGradient is structured like an RBM itself,
// but its values represent the partials of some
// function with respect to an RBM's values.
type RBMGradient RBM

// LogLikelihoodGradient uses contrastive divergence
// to approximate the gradient of the log likelihood
// of the RBM for the given visible inputs.
//
// The markovSteps parameter specifies how many steps
// of Gibbs sampling this should perform for
// contrastive divergence.
func (r *RBM) LogLikelihoodGradient(inputs [][]bool, gibbsSteps int) *RBMGradient {
	grad := RBMGradient(*NewRBM(len(r.VisibleBiases), len(r.HiddenBiases)))

	visibleVec := make(linalg.Vector, len(r.VisibleBiases))

	for _, input := range inputs {
		for i, x := range input {
			if x {
				visibleVec[i] = 1
			} else {
				visibleVec[i] = 0
			}
		}
		grad.VisibleBiases.Add(visibleVec)
		expHidden := r.ExpectedHidden(input)
		grad.HiddenBiases.Add(expHidden)
		for hiddenIdx := 0; hiddenIdx < grad.Weights.Rows; hiddenIdx++ {
			for visibleIdx := 0; visibleIdx < grad.Weights.Cols; visibleIdx++ {
				val := grad.Weights.Get(hiddenIdx, visibleIdx)
				val += expHidden[hiddenIdx] * visibleVec[visibleIdx]
				grad.Weights.Set(hiddenIdx, visibleIdx, val)
			}
		}
	}

	contrastiveDivergence(r, &grad, len(inputs), gibbsSteps)

	return &grad
}

func contrastiveDivergence(r *RBM, grad *RBMGradient, sampleCount int, steps int) {
	visibleState := make([]bool, len(r.VisibleBiases))
	hiddenState := make([]bool, len(r.HiddenBiases))
	for i := 0; i < steps; i++ {
		r.ExpectedHidden(visibleState)
		r.ExpectedVisible(hiddenState)
	}

	scaler := float64(sampleCount)
	visibleVec := make(linalg.Vector, len(visibleState))
	hiddenVec := make(linalg.Vector, len(hiddenState))
	for i, v := range visibleState {
		if v {
			visibleVec[i] = scaler
		}
	}
	for i, h := range hiddenState {
		if h {
			hiddenVec[i] = scaler
		}
	}

	grad.HiddenBiases.Add(hiddenVec)
	grad.VisibleBiases.Add(visibleVec)

	for hiddenIdx := 0; hiddenIdx < grad.Weights.Rows; hiddenIdx++ {
		for visibleIdx := 0; visibleIdx < grad.Weights.Cols; visibleIdx++ {
			val := grad.Weights.Get(hiddenIdx, visibleIdx)
			val += hiddenVec[hiddenIdx] + visibleVec[visibleIdx]
			grad.Weights.Set(hiddenIdx, visibleIdx, val)
		}
	}
}
