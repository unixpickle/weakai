package rbf

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
)

func TestDistLayerOutput(t *testing.T) {
	net := NewDistLayer(10, 3, 0)
	net.SetCenters([]linalg.Vector{
		{1, 2, 1, 2, 1, 2, 1, 2, 2, 1},
		{1, 0, 0, 0, 2, 1, 0, 0, 1, 0},
		{0, 0, 0, 0, 1, 0, 0, 0, 1, 1},
	})
	in := &autofunc.Variable{Vector: []float64{1, 1, 1, 1, 2, 1, 1, 0, 1, 2}}
	actual := net.Apply(in).Output()
	expected := []float64{10, 8, 8}
	if actual.Copy().Scale(-1).Add(expected).MaxAbs() > 1e-6 {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}

func TestDistLayerProp(t *testing.T) {
	net := NewDistLayer(10, 3, 0)
	net.SetCenters([]linalg.Vector{
		{1, 2, 1, 2, 1, 2, 1, 2, 2, 1},
		{1, 0, 0, 0, 2, 1, 0, 0, 1, 0},
		{0, 0, 0, 0, 1, 0, 0, 0, 1, 1},
	})
	in := &autofunc.Variable{Vector: []float64{1, 1, 1, 1, 2, 1, 1, 0, 1, 2}}
	vars := append(net.Parameters(), in)
	rv := autofunc.RVector{}
	for _, v := range vars {
		rv[v] = make(linalg.Vector, len(v.Vector))
		for i := range rv[v] {
			rv[v][i] = rand.NormFloat64()
		}
	}
	checker := functest.RFuncChecker{
		F:     net,
		Vars:  vars,
		Input: in,
		RV:    rv,
	}
	checker.FullCheck(t)
}

func BenchmarkDistLayer(b *testing.B) {
	f := NewDistLayer(300, 300, 1)
	in := &autofunc.Variable{Vector: make(linalg.Vector, 300)}
	for i := range in.Vector {
		in.Vector[i] = rand.NormFloat64()
	}
	g := autofunc.NewGradient(append(f.Parameters(), in))
	up := append(linalg.Vector{}, in.Vector...)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f.Apply(in).PropagateGradient(append(linalg.Vector{}, up...), g)
	}
}
