package rbf

import (
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
)

func TestExpLayerOutput(t *testing.T) {
	l := &ExpLayer{}
	in := &autofunc.Variable{Vector: []float64{-2, -3, 1, -1}}
	actual := l.Apply(in).Output()
	expected := autofunc.Exp{}.Apply(in).Output()
	if actual.Copy().Scale(-1).Add(expected).MaxAbs() > 1e-5 {
		t.Errorf("expected %v but got %v", expected, actual)
	}

	l.Normalize = true
	actual = l.Apply(in).Output()
	expOut := autofunc.Exp{}.Apply(in)
	expSum := autofunc.SumAll(expOut)
	expected = autofunc.ScaleFirst(expOut, autofunc.Inverse(expSum)).Output()
	if actual.Copy().Scale(-1).Add(expected).MaxAbs() > 1e-5 {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}

func TestExpLayerProp(t *testing.T) {
	l := &ExpLayer{Normalize: true}
	in := &autofunc.Variable{Vector: []float64{-2, -3, 1, -1}}
	rv := autofunc.RVector{in: linalg.RandVector(4)}
	ch := functest.RFuncChecker{
		F:     l,
		Input: in,
		Vars:  []*autofunc.Variable{in},
		RV:    rv,
	}
	ch.FullCheck(t)
}
