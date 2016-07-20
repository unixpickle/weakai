package idtrees

import "testing"

type treeTestSample map[string]interface{}

func (t treeTestSample) Attr(n string) interface{} {
	return t[n]
}

func (t treeTestSample) Class() interface{} {
	return t["class"]
}

type treeTest struct {
	Samples  []Sample
	Attrs    []string
	Expected *Tree
}

func (t *treeTest) Run(test *testing.T, prefix string) {
	for maxGos := 0; maxGos < 4; maxGos++ {
		actual := t.actual(maxGos)
		if !treesEqual(t.Expected, actual) {
			test.Errorf(prefix+": bad tree with %d Gos:\n%s", maxGos, actual.String())
			break
		}
	}
}

func (t *treeTest) actual(maxGos int) *Tree {
	return ID3(t.Samples, t.Attrs, maxGos)
}

func treesEqual(t1, t2 *Tree) bool {
	if t1.Classification != nil {
		if t2.Classification == nil {
			return false
		}
		if len(t1.Classification) != len(t2.Classification) {
			return false
		}
		for k, v := range t1.Classification {
			if t2.Classification[k] != v {
				return false
			}
		}
		return true
	}

	if t1.Attr != t2.Attr {
		return false
	}

	if t1.NumSplit != nil {
		if t2.NumSplit == nil {
			return false
		}
		if t1.NumSplit.Threshold != t2.NumSplit.Threshold {
			return false
		}
		return treesEqual(t1.NumSplit.Greater, t2.NumSplit.Greater) &&
			treesEqual(t1.NumSplit.LessEqual, t2.NumSplit.LessEqual)
	}

	if len(t1.ValSplit) != len(t2.ValSplit) {
		return false
	}
	for val, tree := range t1.ValSplit {
		tree2 := t2.ValSplit[val]
		if tree2 == nil {
			return false
		}
		if !treesEqual(tree, tree2) {
			return false
		}
	}

	return true
}
