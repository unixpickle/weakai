package softmax

import "github.com/unixpickle/num-analysis/linalg"

type Gradient []linalg.Vector

func (g Gradient) Inputs() []linalg.Vector {
	return g
}

func (g Gradient) Params() []linalg.Vector {
	return nil
}
