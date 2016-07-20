package idtrees

import "math/rand"

// A TreeGen generates decision trees which classify
// a set of samples using a set of attributes.
type TreeGen func(s []Sample, attrs []string) *Tree

// A Forest is a list of bagged trees that are used
// to classify samples.
type Forest []*Tree

// BuildForest creates a random forest with n trees,
// where each tree was trained on nSamples samples
// and nAttrs attributes.
//
// If nAttrs is 0, the rounded square root of the
// number of attributes is used.
func BuildForest(n int, samples []Sample, attrs []string,
	nSamples, nAttrs int, g TreeGen) Forest {
	if nAttrs == 0 {
		nAttrs = int(float64(len(attrs)) + 0.5)
	}
	sampleCopy := make([]Sample, len(samples))
	attrCopy := make([]string, len(attrs))

	copy(sampleCopy, samples)
	copy(attrCopy, attrs)

	res := make(Forest, n)
	for i := 0; i < n; i++ {
		randomizeSamples(sampleCopy, nSamples)
		randomizeAttrs(attrCopy, nAttrs)
		res[i] = g(sampleCopy[:nSamples], attrCopy[:nAttrs])
	}
	return res
}

// Classify uses f to compute the class probabilities
// of the given sample.
func (f Forest) Classify(s AttrMap) map[interface{}]float64 {
	res := map[interface{}]float64{}
	for _, t := range f {
		x := t.Classify(s)
		for k, v := range x {
			res[k] += v
		}
	}
	scaler := 1 / float64(len(f))
	for k, v := range res {
		res[k] = v * scaler
	}
	return res
}

func randomizeSamples(s []Sample, n int) {
	for i := 0; i < n; i++ {
		idx := rand.Intn(len(s)-i) + i
		s[i], s[idx] = s[idx], s[i]
	}
}

func randomizeAttrs(a []string, n int) {
	for i := 0; i < n; i++ {
		idx := rand.Intn(len(a)-i) + i
		a[i], a[idx] = a[idx], a[i]
	}
}
